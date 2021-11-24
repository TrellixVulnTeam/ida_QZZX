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

from liga.interpret.common import Interpreter
from liga.liga import liga, Resampler
from liga.torch_extensions.classifier import TorchImageClassifier
from liga.type1.common import Type1Explainer, top_k_accuracy_score, counterfactual_top_k_accuracy_score
from liga.common import NestedLogger


# See https://stackoverflow.com/questions/15063936
max_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_field_size)
        break
    except OverflowError:
        maxInt = int(max_field_size / 10)


@contextmanager
def _prepare_image_iter(images_url, skip: int = 0) -> Iterable[Tuple[str, np.ndarray]]:
    reader = make_reader(images_url, workers_count=1)  # only 1 worker to ensure determinism of results
    try:
        def image_iter() -> Iterable[Tuple[str, np.ndarray]]:
            for row in it.islice(reader, skip, None):
                yield row.image_id, row.image

        yield image_iter()
    finally:
        reader.stop()
        reader.join()


_TEST_CACHE = {}


def _prepare_test_observations(test_images_url: str,
                               interpreter: Interpreter,
                               classifier: TorchImageClassifier,
                               num_test_obs: int) -> Tuple[List[Tuple[str, np.ndarray]],
                                                           List[List[int]],
                                                           List[int]]:
    key = (test_images_url, interpreter, classifier.name, num_test_obs)
    if key not in _TEST_CACHE:
        images = []
        concept_counts = []
        predicted_classes = []
        with _prepare_image_iter(test_images_url) as test_iter:
            for image_id, image in it.islice(test_iter, num_test_obs):
                images.append((image_id, image))
                ids_and_masks = list(interpreter(image, image_id))
                if len(ids_and_masks) > 0:
                    concept_ids, _ = list(zip(*interpreter(image, image_id)))
                    concept_counts.append(interpreter.concept_ids_to_counts(concept_ids))
                else:
                    concept_counts.append(np.zeros(len(interpreter.concepts), dtype=np.int))
                predicted_classes.append(classifier.predict_single(image))
            _TEST_CACHE[key] = images, concept_counts, predicted_classes

    return _TEST_CACHE[key]


@dataclass
class Experiment(NestedLogger):
    rng: np.random.Generator
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
    resampler: Resampler
    top_k_acc: int = 5

    def __post_init__(self):
        assert len(self.all_classes) == self.classifier.num_classes, \
            'Provided number of class names differs from the output dimensionality of the classifier.'
        assert self.num_test_obs_for_counterfactuals <= self.num_test_obs, \
            'Number of counterfactual test observations cannot exceed the number of regular test observations.'

    @property
    def classifier(self) -> TorchImageClassifier:
        return self.resampler.classifier

    @property
    def interpreter(self) -> Interpreter:
        return self.resampler.interpreter

    @property
    def params(self) -> Mapping[str, Any]:
        return {'classifier': self.classifier.name,
                'images_url': self.images_url,
                'num_train_obs': self.num_train_obs,
                'num_calibration_obs': self.num_calibration_obs,
                'interpreter': str(self.interpreter),
                'type1': str(self.type1),
                'resampler': str(self.resampler),
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
        from the LIGA algorithm, the hyperparameters of the fitted surrogate model and the experimental results.
        Passes *kwargs* to the LIGA algorithm.
        """
        with self.log_task('Running experiment...'):
            self.log_item('Parameters: {}'.format(self.params))

            with self.log_task('Caching test observations...'):
                self.images_test, self.counts_test, self.y_test = _prepare_test_observations(self.images_url,
                                                                                             self.interpreter,
                                                                                             self.classifier,
                                                                                             self.num_test_obs)

            with _prepare_image_iter(self.images_url, skip=self.num_test_obs) as train_images_iter:
                for rep_no in range(1, self.repetitions + 1):
                    calibration_images_iter = it.islice(train_images_iter, self.num_calibration_obs)
                    with self.log_task('Calibrating the resampler...'):
                        self.resampler.calibrate(calibration_images_iter)

                    # we draw new train observations for each repetition by continuing the iterator
                    start = timeit.default_timer()
                    surrogate, stats = liga(rng=self.rng,
                                            type1=self.type1,
                                            resampler=self.resampler,
                                            image_iter=it.islice(train_images_iter,
                                                                 self.num_train_obs),
                                            log_nesting=self.log_nesting,
                                            **kwargs)
                    stop = timeit.default_timer()

                    with self.log_task('Scoring surrogate model...'):
                        fit_params, metrics = self.score(surrogate)

                    metrics['runtime_s'] = stop - start
                    yield surrogate, stats, fit_params, metrics, self.type1.get_plot_representation(surrogate)

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

    def score(self, surrogate) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        y_test_pred = surrogate.predict_proba(self.counts_test)
        y_test_pred = self._spread_probs_to_all_classes(y_test_pred, surrogate.classes_)

        if y_test_pred.shape[1] == 2:
            # binary case
            y_test_pred = y_test_pred[:, 1]

        fit_params = self.type1.get_fitted_params(surrogate)
        metrics = self.type1.get_complexity_metrics(surrogate)
        metrics.update({'cross_entropy': log_loss(self.y_test, y_test_pred, labels=self.all_class_ids),
                        'auc': self._auc_score(self.y_test, y_test_pred),
                        'acc': top_k_accuracy_score(estimator=surrogate,
                                                    inputs=self.counts_test,
                                                    target_outputs=self.y_test,
                                                    k=self.top_k_acc,
                                                    all_classes=self.all_classes,
                                                    logger=self),
                        'counterfactual_acc': self._score_counterfactual_accuracy(surrogate)})
        return fit_params, metrics

    def _auc_score(self, y_true, y_pred):
        multi_class = 'ovo' if self.multi_class else 'raise'
        auc = roc_auc_score(y_true,
                            y_pred,
                            average='macro',
                            multi_class=multi_class,
                            labels=self.all_class_ids)
        return auc

    def _score_counterfactual_accuracy(self, surrogate):
        """
        This score is different from "normal" accuracy, because it requires that predictions
        are correct for a pair of an input and its "counterfactual twin" where a concept has been removed.
        """
        logger = NestedLogger()
        logger.log_nesting = self.log_nesting
        return counterfactual_top_k_accuracy_score(surrogate=surrogate,
                                                   images=self.images_test,
                                                   counts=self.counts_test,
                                                   target_classes=self.y_test,
                                                   classifier=self.classifier,
                                                   interpreter=self.interpreter,
                                                   max_perturbed_area=self.max_perturbed_area,
                                                   max_concept_overlap=self.max_concept_overlap,
                                                   k=self.top_k_acc,
                                                   logger=logger,
                                                   all_classes=self.all_classes)


def get_results_dir() -> ContextManager[Path]:
    return resources.path('liga.experiments', 'results')


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
        fields = ['exp_no', 'params', 'rep_no', 'stats', 'fit_params', 'metrics', 'plot_repr']
        packed_writer = csv.DictWriter(packed_csv_file, fieldnames=fields)
        packed_writer.writeheader()

        # first_run = True

        for exp_no, e in enumerate(experiments, start=1):
            for rep_no, (surrogate, stats, fit_params, metrics, plot_repr) in enumerate(e.run(**kwargs)):
                packed_writer.writerow({'exp_no': exp_no,
                                        'params': e.params,
                                        'rep_no': rep_no,
                                        'stats': stats,
                                        'fit_params': fit_params,
                                        'metrics': metrics,
                                        'plot_repr': plot_repr})
                packed_csv_file.flush()  # make sure intermediate results are written


def get_experiment_df(experiment_name: str) -> pd.DataFrame:
    with resources.path('liga.experiments.results', experiment_name) as path:
        with (path / 'results_packed.csv').open('r') as results_csv_file:
            unpacked_column_names = set()

            # first pass: find out all column names
            unpacked_rows = []
            for row in csv.DictReader(results_csv_file):
                packed = {k: ast.literal_eval(v) for k, v in row.items()}
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
