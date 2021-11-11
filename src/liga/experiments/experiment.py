import base64
import csv
import itertools as it
import logging
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
import timeit
from datetime import datetime
from pathlib import Path
from typing import Mapping, Any, Dict, Tuple, Iterable, List, ContextManager

import numpy as np
from importlib import resources

from petastorm import make_reader
from sklearn.metrics import log_loss, roc_auc_score

from liga.liga import liga, concept_ids_to_counts
from liga.torch_extensions.classifier import TorchImageClassifier
from liga.type1.common import Type1Explainer
from liga.type2.common import Type2Explainer
from simexp.common import NestedLogger


@dataclass
class Experiment(NestedLogger):
    rng: np.random.Generator
    repetitions: int
    train_images_url: str
    test_images_url: str
    num_train_obs: int
    num_test_obs: int
    all_classes: [str]
    type1: Type1Explainer
    type2: Type2Explainer
    top_k_acc: int = 5

    def __post_init__(self):
        assert len(self.all_classes) == self.classifier.num_classes, \
            'Provided number of class names differs from the output dimensionality of the classifier.'

    @property
    def classifier(self) -> TorchImageClassifier:
        return self.type2.classifier

    @property
    def concepts(self) -> [str]:
        return self.type2.interpreter.concepts

    @property
    def params(self) -> Mapping[str, Any]:
        return {'classifier': self.classifier.name,
                'images_url': self.train_images_url,
                'num_train_obs': self.num_train_obs,
                'interpreter': str(self.type2.interpreter),
                'type1': str(self.type1),
                'type2': str(self.type2),
                'num_test_obs': self.num_test_obs}

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
        with self.log_task('Generating and caching test observations...'):
            x_test, y_test = self._prepare_test_observations()

        with self.log_task('Running experiment...'):
            self.log_item('Parameters: {}'.format(self.params))

            with self._prepare_image_iter(self.train_images_url) as train_iter:
                for rep_no in range(1, self.repetitions + 1):
                    # we draw new train and test observations for each repetition
                    train_obs = it.islice(train_iter, self.num_train_obs)
                    start = timeit.default_timer()
                    surrogate, stats = liga(rng=self.rng,
                                            type1=self.type1,
                                            type2=self.type2,
                                            image_iter=train_obs,
                                            log_nesting=self.log_nesting,
                                            **kwargs)
                    stop = timeit.default_timer()

                    with self.log_task('Scoring surrogate model...'):
                        fit_params, metrics = self.score(surrogate, x_test, y_test)

                    metrics['runtime_s'] = stop - start
                    yield surrogate, stats, fit_params, metrics

    @contextmanager
    def _prepare_image_iter(self, images_url) -> Iterable[Tuple[str, np.ndarray]]:
        reader = make_reader(images_url,
                             workers_count=1,  # only 1 worker to ensure determinism of results
                             shuffle_row_groups=True)  # this is non-deterministic unless you set random.seed() !
        try:
            def image_iter() -> Iterable[Tuple[str, np.ndarray]]:
                for row in reader:
                    yield row.image_id, row.image

            yield image_iter()
        finally:
            reader.stop()
            reader.join()

    def _prepare_test_observations(self) -> Tuple[List[List[int]], List[int]]:
        concept_counts = []
        predicted_classes = []
        with self._prepare_image_iter(self.test_images_url) as test_iter:
            for image_id, image in test_iter:
                ids_and_masks = list(self.type2.interpreter(image, image_id))
                if len(ids_and_masks) > 0:
                    concept_ids, _ = list(zip(*self.type2.interpreter(image, image_id)))
                    concept_counts.append(concept_ids_to_counts(concept_ids,
                                                                len(self.concepts)))
                else:
                    concept_counts.append(np.zeros(len(self.concepts), dtype=np.int))
                predicted_classes.append(self.classifier.predict_single(image))
            return concept_counts, predicted_classes

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

    def score(self, surrogate, x_test, y_test) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        y_test_pred = surrogate.predict_proba(x_test)
        y_test_pred = self._spread_probs_to_all_classes(y_test_pred, surrogate.classes_)

        if y_test_pred.shape[1] == 2:
            # binary case
            y_test_pred = y_test_pred[:, 1]

        fit_params = self.type1.get_fitted_params(surrogate)
        metrics = self.type1.get_complexity_metrics(surrogate)
        metrics.update({'cross_entropy': log_loss(y_test, y_test_pred, labels=self.all_class_ids),
                        'gini': self._gini_score(y_test, y_test_pred),
                        'acc': self._top_k_acc_score(surrogate, x_test, y_test)})
        return fit_params, metrics

    def _top_k_acc_score(self, estimator, inputs, target_outputs):
        cls_ids = {cls: cls_id for cls_id, cls in enumerate(estimator.classes_)}
        predicted = estimator.predict_proba(inputs)
        top_k_indices = np.argsort(predicted)[:, -self.top_k_acc:]
        warnings = set()

        hit_count = 0
        for y, pred_indices in zip(target_outputs, top_k_indices):
            try:
                cls_id = cls_ids[y]
            except KeyError:
                warnings.add(self.all_classes[y])
            else:
                if cls_id in pred_indices:
                    hit_count += 1

        if warnings:
            self.log_item('Classes {} of the test data were not in the training data of this classifier.'
                          .format(warnings))

        return float(hit_count) / float(len(inputs))

    def _gini_score(self, y_true, y_pred):
        multi_class = 'ovo' if self.multi_class else 'raise'
        auc = roc_auc_score(y_true,
                            y_pred,
                            average='macro',
                            multi_class=multi_class,
                            labels=self.all_class_ids)
        return 2. * auc - 1.


def get_results_dir() -> ContextManager[Path]:
    return resources.path('liga.experiments', 'results')


def encode_surrogate(surrogate):
    return base64.b64encode(pickle.dumps(surrogate))


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
        with (exp_dir / 'results_unpacked.csv').open('w') as unpacked_csv_file:

            fields = ['params', 'rep_no', 'stats', 'fit_params', 'metrics', 'surrogate_obj']
            packed_writer = csv.DictWriter(packed_csv_file, fieldnames=fields)
            packed_writer.writeheader()

            first_run = True

            for e in experiments:
                for rep_no, (surrogate, stats, fit_params, metrics) in enumerate(e.run(**kwargs)):
                    surrogate_str = encode_surrogate(surrogate)
                    packed_writer.writerow({'params': e.params,
                                            'rep_no': rep_no,
                                            'stats': stats,
                                            'fit_params': fit_params,
                                            'metrics': metrics,
                                            'surrogate_obj': surrogate_str})
                    packed_csv_file.flush()  # make sure intermediate results are written

                    if first_run:
                        param_names = list(e.params.keys())
                        stats_names = list(stats.keys())
                        fit_params_names = list(fit_params.keys())
                        metric_names = list(metrics.keys())
                        all_names = (set(param_names)
                                     .union(set(metric_names))
                                     .union(set(fit_params_names))
                                     .union(set(stats_names)))
                        exp_names_len = len(param_names) + len(stats_names) + len(fit_params_names) + len(metric_names)
                        can_unpack = len(all_names) == exp_names_len
                        fields = (param_names +
                                  ['rep_no'] +
                                  stats_names +
                                  fit_params_names +
                                  metric_names +
                                  ['surrogate_obj'])
                        unpacked_writer = csv.DictWriter(unpacked_csv_file, fieldnames=fields)
                        unpacked_writer.writeheader()
                        first_run = False

                    if set(e.params.keys()) == set(param_names) \
                            and set(metrics.keys()) == set(metric_names) \
                            and set(fit_params.keys()) == set(fit_params_names) \
                            and set(stats.keys()) == set(stats_names) \
                            and can_unpack:
                        merged = {**e.params,
                                  **metrics,
                                  **stats,
                                  **fit_params,
                                  'rep_no': rep_no,
                                  'surrogate_obj': surrogate_str}
                        unpacked_writer.writerow(merged)
                        unpacked_csv_file.flush()
                    else:
                        logging.warning('Could not unpack result of experiment into the same csv as previous results,'
                                        'because the experiment generates a different set of params, '
                                        'stats or metrics.\n'
                                        'Params: {}\n'
                                        'Stats: {}\n'
                                        'Metrics: {}'.format(e.params, stats, metrics))
