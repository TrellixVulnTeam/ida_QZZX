import csv
import itertools as it
import logging
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
from liga.type1.common import Type1Explainer
from liga.common import NestedLogger


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
                               num_test_obs: int) -> Tuple[Iterable[List[int]], Iterable[int]]:
    key = (test_images_url, interpreter, classifier.name, num_test_obs)
    if key not in _TEST_CACHE:
        concept_counts = []
        predicted_classes = []
        with _prepare_image_iter(test_images_url) as test_iter:
            for image_id, image in it.islice(test_iter, num_test_obs):
                ids_and_masks = list(interpreter(image, image_id))
                if len(ids_and_masks) > 0:
                    concept_ids, _ = list(zip(*interpreter(image, image_id)))
                    concept_counts.append(interpreter.concept_ids_to_counts(concept_ids))
                else:
                    concept_counts.append(np.zeros(len(interpreter.concepts), dtype=np.int))
                predicted_classes.append(classifier.predict_single(image))
            _TEST_CACHE[key] = concept_counts, predicted_classes

    return _TEST_CACHE[key]


@dataclass
class Experiment(NestedLogger):
    rng: np.random.Generator
    repetitions: int
    images_url: str
    num_train_obs: int
    num_test_obs: int
    num_test_obs_for_counterfactuals: int
    num_counterfactuals: int
    max_perturbed_area: float
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
                'interpreter': str(self.interpreter),
                'type1': str(self.type1),
                'resampler': str(self.resampler),
                'num_test_obs': self.num_test_obs,
                'num_test_obs_for_counterfactuals': self.num_test_obs_for_counterfactuals,
                'num_counterfactuals': self.num_counterfactuals,
                'max_perturbed_area': self.max_perturbed_area}

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
                self.x_test, self.y_test = _prepare_test_observations(self.images_url,
                                                                      self.interpreter,
                                                                      self.classifier,
                                                                      self.num_test_obs)

            with _prepare_image_iter(self.images_url, skip=self.num_test_obs) as train_images_iter:
                for rep_no in range(1, self.repetitions + 1):
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
        y_test_pred = surrogate.predict_proba(self.x_test)
        y_test_pred = self._spread_probs_to_all_classes(y_test_pred, surrogate.classes_)

        if y_test_pred.shape[1] == 2:
            # binary case
            y_test_pred = y_test_pred[:, 1]

        fit_params = self.type1.get_fitted_params(surrogate)
        metrics = self.type1.get_complexity_metrics(surrogate)
        metrics.update({'cross_entropy': log_loss(self.y_test, y_test_pred, labels=self.all_class_ids),
                        'gini': self._gini_score(self.y_test, y_test_pred),
                        'acc': self._top_k_acc_score(surrogate, self.x_test, self.y_test),
                        'counterfactual_acc': self._score_counterfactual_accuracy(surrogate)})
        return fit_params, metrics

    def _top_k_acc_score(self, estimator, inputs, target_outputs):
        cls_ids = {cls: cls_id for cls_id, cls in enumerate(estimator.classes_)}
        predicted = estimator.predict_proba(inputs)
        top_k_indices = np.argsort(predicted)[:, -self.top_k_acc:]
        warnings = set()

        total_count = 0
        hit_count = 0
        for y, pred_indices in zip(target_outputs, top_k_indices):
            total_count += 1
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

        return float(hit_count) / float(total_count)

    def _gini_score(self, y_true, y_pred):
        multi_class = 'ovo' if self.multi_class else 'raise'
        auc = roc_auc_score(y_true,
                            y_pred,
                            average='macro',
                            multi_class=multi_class,
                            labels=self.all_class_ids)
        return 2. * auc - 1.

    def _score_counterfactual_accuracy(self, surrogate):
        pair_count = 0.
        faithful_count = 0.

        with _prepare_image_iter(self.images_url) as test_iter:
            for image_id, image in it.islice(test_iter, self.num_test_obs_for_counterfactuals):
                ids_and_masks = list(self.interpreter(image_id=image_id,
                                                      image=image))
                if len(ids_and_masks) == 0:
                    continue

                true_class = self.classifier.predict_single(image_id=image_id,
                                                            image=image)
                ids, masks = list(zip(*ids_and_masks))
                counts = self.interpreter.concept_ids_to_counts(ids)
                pred_class = surrogate.predict([counts])[0]

                pair_count += float(self.num_counterfactuals)

                if true_class == pred_class:
                    # this is different from "normal" accuracy, because it requires that predictions
                    # are correct for a pair of an input and its counterfactual "twin" where some concepts are removed
                    cf_iter = self.interpreter.get_counterfactuals(self.rng,
                                                                   image=image,
                                                                   image_id=image_id,
                                                                   max_counterfactuals=self.num_counterfactuals,
                                                                   max_perturbed_concepts=1,
                                                                   max_perturbed_area=self.max_perturbed_area)
                    for cf_counts, cf_image in cf_iter:
                        cf_true_class = self.classifier.predict_single(image=cf_image)
                        cf_pred = surrogate.predict([cf_counts])[0]
                        if cf_true_class == cf_pred:
                            faithful_count += 1.

        return faithful_count / pair_count


def powerset(iterable, include_empty_set=True):
    start = 0 if include_empty_set else 1
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(start, len(s) + 1))


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
        with (exp_dir / 'results_unpacked.csv').open('w') as unpacked_csv_file:

            fields = ['exp_no', 'params', 'rep_no', 'stats', 'fit_params', 'metrics', 'plot_repr']
            packed_writer = csv.DictWriter(packed_csv_file, fieldnames=fields)
            packed_writer.writeheader()

            first_run = True

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

                    if first_run:
                        param_names = list(e.params.keys())
                        stats_names = list(stats.keys())
                        fit_params_names = list(fit_params.keys())
                        metric_names = list(metrics.keys())
                        plot_repr_names = list(plot_repr.keys())
                        all_names = (set(param_names)
                                     .union(set(metric_names))
                                     .union(set(fit_params_names))
                                     .union(set(stats_names))
                                     .union(set(plot_repr_names)))
                        exp_names_len = len(param_names) + len(stats_names) + len(fit_params_names) \
                            + len(metric_names) + len(plot_repr_names)
                        can_unpack = len(all_names) == exp_names_len
                        fields = (['exp_no'] +
                                  param_names +
                                  ['rep_no'] +
                                  stats_names +
                                  fit_params_names +
                                  metric_names +
                                  plot_repr_names)
                        unpacked_writer = csv.DictWriter(unpacked_csv_file, fieldnames=fields)
                        unpacked_writer.writeheader()
                        first_run = False

                    if set(e.params.keys()) == set(param_names) \
                            and set(metrics.keys()) == set(metric_names) \
                            and set(fit_params.keys()) == set(fit_params_names) \
                            and set(stats.keys()) == set(stats_names) \
                            and set(plot_repr.keys()) == set(plot_repr_names) \
                            and can_unpack:
                        merged = {**e.params,
                                  **metrics,
                                  **stats,
                                  **fit_params,
                                  **plot_repr,
                                  'rep_no': rep_no,
                                  'exp_no': exp_no}
                        unpacked_writer.writerow(merged)
                        unpacked_csv_file.flush()
                    else:
                        logging.warning('Could not unpack result of experiment into the same csv as '
                                        'previous results, because the experiment generates a different set '
                                        'of params, stats or metrics.\n'
                                        'Params: {}\n'
                                        'Stats: {}\n'
                                        'Metrics: {}'.format(e.params, stats, metrics))


def get_experiment_df(experiment_name: str) -> pd.DataFrame:
    with resources.path('liga.experiments.results', experiment_name) as path:
        return pd.read_csv(path / 'results_unpacked.csv')