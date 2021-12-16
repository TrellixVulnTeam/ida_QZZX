import ast
import csv
import itertools as it
import sys
from contextlib import contextmanager
from dataclasses import dataclass
import timeit
from datetime import datetime
from pathlib import Path
from typing import Mapping, Any, Dict, Tuple, Iterable, List, ContextManager

import numpy as np
from importlib import resources

import pandas as pd
from petastorm import make_reader
from sklearn.metrics import log_loss, roc_auc_score

from ida.ida import ipa
from ida.interpret.common import Interpreter
from ida.torch_extensions.classifier import TorchImageClassifier
from ida.type1.common import Type1Explainer, top_k_accuracy_score
from ida.common import NestedLogger, memory

# See https://stackoverflow.com/questions/15063936
from ida.type2.common import Type2Explainer

max_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_field_size)
        break
    except OverflowError:
        maxInt = int(max_field_size / 10)


@contextmanager
def _prepare_image_iter(images_url, skip: int = 0) -> Iterable[Tuple[str, np.ndarray]]:
    reader = make_reader(images_url,
                         workers_count=1,  # only 1 worker to ensure determinism of results
                         shuffle_row_groups=False)
    try:
        def image_iter() -> Iterable[Tuple[str, np.ndarray]]:
            for row in it.islice(reader, skip, None):
                yield row.image_id, row.image

        yield image_iter()
    finally:
        reader.stop()
        reader.join()


@memory.cache
def _prepare_test_observations(test_images_url: str,
                               interpreter: Interpreter,
                               classifier: TorchImageClassifier,
                               num_test_obs: int,
                               num_cf_test_obs: int,
                               max_perturbed_area: float,
                               max_concept_overlap: float) -> Tuple[List[List[int]], List[int],
                                                                    List[List[int]], List[int]]:
    concept_counts = []
    predicted_classes = []
    cf_concept_counts = []
    cf_predicted_classes = []
    with _prepare_image_iter(test_images_url) as test_iter:
        for image_id, image in it.islice(test_iter, num_test_obs):
            concept_counts.append(interpreter.count_concepts(image=image, image_id=image_id))
            predicted_classes.append(classifier.predict_single(image=image))

        for image_id, image in it.islice(test_iter, num_cf_test_obs):
            cf_concept_counts.append(interpreter.count_concepts(image=image, image_id=image_id))
            cf_predicted_classes.append(classifier.predict_single(image=image))
            for cf_counts, cf_image in interpreter.get_counterfactuals(max_perturbed_area=max_perturbed_area,
                                                                       max_concept_overlap=max_concept_overlap,
                                                                       image_id=image_id,
                                                                       image=image):
                cf_concept_counts.append(cf_counts)
                cf_predicted_classes.append(classifier.predict_single(image=image))

    return concept_counts, predicted_classes, cf_concept_counts, cf_predicted_classes


@dataclass
class Experiment(NestedLogger):
    random_state: int
    repetitions: int
    images_url: str
    num_train_obs: int
    num_calibration_obs: int
    num_test_obs: int
    num_test_obs_for_counterfactuals: int
    max_perturbed_area: float
    max_concept_overlap: float
    all_classes: [str]
    type1: Type1Explainer
    type2: Type2Explainer
    top_k_acc: Iterable[int]

    def __post_init__(self):
        assert len(self.all_classes) == self.classifier.num_classes, \
            'Provided number of class names differs from the output dimensionality of the classifier.'

    @property
    def classifier(self) -> TorchImageClassifier:
        return self.type2.classifier

    @property
    def interpreter(self) -> Interpreter:
        return self.type2.interpreter

    @property
    def params(self) -> Mapping[str, Any]:
        return {'classifier': self.classifier.name,
                'images_url': self.images_url,
                'num_train_obs': self.num_train_obs,
                'num_calibration_obs': self.num_calibration_obs,
                'interpreter': str(self.interpreter),
                'type2': str(self.type2),
                'type1': str(self.type1),
                'num_test_obs': self.num_test_obs,
                'num_test_obs_for_counterfactuals': self.num_test_obs_for_counterfactuals,
                'max_perturbed_area': self.max_perturbed_area,
                'min_overlap_for_concept_merge': self.max_concept_overlap}

    @property
    def multi_class(self) -> bool:
        return self.classifier.num_classes > 2

    @property
    def all_class_ids(self) -> [int]:
        return list(range(self.classifier.num_classes))

    def run(self, **kwargs):
        """
        Runs the configured number of repetitions of the experiment.
        For each repetition, yields a tuple *(surrogate, stats, fit_params, metrics)*.
        *surrogate* is the fitted surrogate model.
        *stats*, *fit_params*, and *metrics* are dicts that contain the augmentation statistics
        from the IDA algorithm, the hyperparameters of the fitted surrogate model and the experimental results.
        Passes *kwargs* to the IPA algorithm.
        """
        with self.log_task('Running experiment...'):
            self.log_item('Parameters: {}'.format(self.params))

            with self.log_task('Caching test observations...'):
                test_obs = _prepare_test_observations(test_images_url=self.images_url,
                                                      interpreter=self.interpreter,
                                                      classifier=self.classifier,
                                                      num_test_obs=self.num_test_obs,
                                                      num_cf_test_obs=self.num_test_obs_for_counterfactuals,
                                                      max_perturbed_area=self.max_perturbed_area,
                                                      max_concept_overlap=self.max_concept_overlap)
                self.counts_test, self.y_test, self.cf_counts_test, self.cf_y_test = test_obs

            with _prepare_image_iter(self.images_url, skip=self.num_test_obs) as train_images_iter:
                for rep_no in range(1, self.repetitions + 1):
                    calibration_images_iter = it.islice(train_images_iter, self.num_calibration_obs)
                    with self.log_task('Calibrating the Type 2 explainer...'):
                        self.type2.calibrate(calibration_images_iter)

                    # we draw new train observations for each repetition by continuing the iterator
                    start = timeit.default_timer()
                    cv = ipa(type1=self.type1,
                             type2=self.type2,
                             image_iter=it.islice(train_images_iter,
                                                  self.num_train_obs),
                             log_nesting=self.log_nesting,
                             random_state=self.random_state,
                             **kwargs)
                    stop = timeit.default_timer()

                    surrogate = cv.best_estimator_['approximate']
                    picker = cv.best_estimator_['interpret-pick']
                    picked_concepts = picker.picked_concepts_
                    with self.log_task('Scoring surrogate model...'):
                        metrics = self.score(surrogate, picked_concepts)

                    metrics['runtime_s'] = stop - start
                    fit_params = cv.best_params_
                    stats = picker.stats_
                    yield surrogate, stats, fit_params, metrics, self.type1.serialize(surrogate)

    def _spread_probs_to_all_classes(self, probs, classes_) -> np.ndarray:
        """
        probs: list of probabilities, output of predict_proba
        classes_: classes that the classifier has seen during training (integer ids)

        Returns a list of probabilities indexed by *all* class ids,
        not only those that the classifier has seen during training.
        See https://stackoverflow.com/questions/30036473/
        """
        proba_ordered = np.zeros((probs.shape[0], len(self.all_classes),), dtype=np.float)
        sorter = np.argsort(self.all_classes)  # http://stackoverflow.com/a/32191125/395857
        idx = sorter[np.searchsorted(self.all_class_ids, classes_, sorter=sorter)]
        proba_ordered[:, idx] = probs
        return proba_ordered

    def _get_predictions(self, surrogate, counts: List[List[int]]):
        pred = surrogate.predict_proba(counts)
        pred = self._spread_probs_to_all_classes(pred, surrogate.classes_)
        if pred.shape[1] == 2:
            # binary case
            pred = pred[:, 1]
        return pred

    def score(self, surrogate, picked_concepts: [int]) -> Dict[str, Any]:
        metrics = self.type1.get_complexity_metrics(surrogate)

        picked_counts_test = np.asarray(self.counts_test)[:, picked_concepts]
        y_test_pred = self._get_predictions(surrogate, picked_counts_test)
        picked_cf_counts_test = np.asarray(self.cf_counts_test)[:, picked_concepts]
        cf_y_test_pred = self._get_predictions(surrogate, picked_cf_counts_test)
        metrics.update({'cross_entropy': log_loss(self.y_test, y_test_pred, labels=self.all_class_ids),
                        'auc': self._auc_score(self.y_test, y_test_pred),
                        'cf_cross_entropy': log_loss(self.cf_y_test, cf_y_test_pred, labels=self.all_class_ids),
                        'cf_auc': self._auc_score(self.cf_y_test, cf_y_test_pred)})
        for k in self.top_k_acc:
            metrics.update({f'top_{k}_acc': top_k_accuracy_score(surrogate=surrogate,
                                                                 counts=picked_counts_test,
                                                                 target_classes=self.y_test,
                                                                 k=k),
                            f'cf_top_{k}_acc': top_k_accuracy_score(surrogate=surrogate,
                                                                    counts=picked_cf_counts_test,
                                                                    target_classes=self.cf_y_test,
                                                                    k=k)})
        return metrics

    def _auc_score(self, y_true, y_pred):
        multi_class = 'ovo' if self.multi_class else 'raise'
        auc = roc_auc_score(y_true,
                            y_pred,
                            average='macro',
                            multi_class=multi_class,
                            labels=self.all_class_ids)
        return auc


def get_results_dir() -> ContextManager[Path]:
    return resources.path('ida.experiments', 'results')


def run_experiments(name: str,
                    description: str,
                    experiments: Iterable[Experiment],
                    prepend_timestamp: bool = True,
                    **kwargs):
    if prepend_timestamp:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        name = timestamp + ' ' + name

    with get_results_dir() as results_dir:
        exp_dir = results_dir / name
        exp_dir.mkdir(exist_ok=False)
        with (exp_dir / 'description').open('w') as description_file:
            description_file.write(description)

    with (exp_dir / 'results_packed.csv').open('w') as packed_csv_file:
        fields = ['exp_no', 'params', 'rep_no', 'stats', 'fit_params', 'metrics', 'surrogate_serial']
        packed_writer = csv.DictWriter(packed_csv_file, fieldnames=fields)
        packed_writer.writeheader()

        for exp_no, e in enumerate(experiments, start=1):
            for rep_no, (surrogate, stats, fit_params, metrics, serial) in enumerate(e.run(**kwargs), start=1):
                packed_writer.writerow({'exp_no': exp_no,
                                        'params': e.params,
                                        'rep_no': rep_no,
                                        'stats': stats,
                                        'fit_params': fit_params,
                                        'metrics': metrics,
                                        'surrogate_serial': serial})
                packed_csv_file.flush()  # make sure intermediate results are written


def get_experiment_df(experiment_name: str) -> pd.DataFrame:
    with resources.path('ida.experiments.results', experiment_name) as path:
        with (path / 'results_packed.csv').open('r') as results_csv_file:
            unpacked_column_names = set()

            # first pass: find out all column names
            unpacked_rows = []
            for row in csv.DictReader(results_csv_file):
                packed = {}
                for k, v in row.items():
                    try:
                        packed[k] = ast.literal_eval(v)
                    except ValueError:
                        pass

                unpacked = {}
                for k, v in packed.items():
                    if isinstance(v, dict):
                        unpacked_column_names.update(v.keys())
                        unpacked.update(v)
                    else:
                        unpacked.update({k: v})
                unpacked_rows.append(unpacked)

        unified_rows = []
        for row in unpacked_rows:
            d = {c: None for c in unpacked_column_names}
            d.update(row)
            unified_rows.append(d)

        return pd.DataFrame(unified_rows)
