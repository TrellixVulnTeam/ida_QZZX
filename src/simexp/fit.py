import itertools as it
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import total_ordering
from typing import Union, Any, Dict, Iterable, Tuple, Optional, List

import numpy as np
import pyspark.sql.functions as sf
from petastorm.etl.dataset_metadata import get_schema_from_dataset_url
from petastorm.unischema import Unischema, UnischemaField
from pyspark.sql import DataFrame
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree

from simexp.common import LoggingMixin, ComposableDataclass
from simexp.spark import Field, SparkSessionConfig


@dataclass
class TreeSurrogate(LoggingMixin):

    @total_ordering
    @dataclass
    class Score:
        cv_results: Dict[str, Any]
        cv_best_index: int
        cv_n_splits: int
        tree: Tree

        cross_entropy: float
        gini: float
        top_k_acc: int
        acc: float
        counts: np.ndarray

        def __str__(self):
            res = self.cv_results
            entropies = np.asarray([res['split{}_test_score'.format(i)][self.cv_best_index]
                                    for i in range(self.cv_n_splits)])

            s = 'Best model has\n' \
                '-> Mean Training Cross-entropy: {}\n' \
                '-> Training StdDev: {}\n' \
                '-> Params:\n' \
                .format(np.mean(entropies), np.std(entropies))

            best_params = self.cv_results['params'][self.cv_best_index]
            for k, v in best_params.items():
                s += '    \'{}\': {}\n'.format(k, v)

            s += 'Expected cross-entropy: {:.3f}\n'.format(self.cross_entropy)
            s += 'Expected gini: {:.3f}\n'.format(self.gini)
            s += 'Expected top-{}-acc: {:.3f}\n'.format(self.top_k_acc, self.acc)
            return s

        def __eq__(self, other):
            return self.cross_entropy == other.cross_entropy

        def __lt__(self, other):
            return self.cross_entropy < other.cross_entropy

    param_grid: Dict[str, Any]
    random_state: int
    all_classes: [str]

    k_folds: int = 5
    top_k_acc: int = 5
    n_jobs: int = 62

    def __post_init__(self):
        self.pipeline = Pipeline([
            ('fsel', SelectFromModel(ExtraTreesClassifier(random_state=self.random_state))),
            ('clf', DecisionTreeClassifier(random_state=self.random_state))
        ])
        assert len(self.all_classes) > 1
        self.multi_class = len(self.all_classes) > 2

    @property
    def all_class_ids(self):
        return list(range(len(self.all_classes)))

    def fit_and_score(self, X_train, y_train, X_test, y_test) -> Union[GridSearchCV, Score]:
        search = GridSearchCV(self.pipeline, self.param_grid,
                              cv=self.k_folds, n_jobs=self.n_jobs,
                              scoring='neg_log_loss')
        cv = search.fit(X_train, y_train)

        try:
            return self._score(cv, X_test, y_test)
        except Exception as e:
            self._log_item('ERROR: The following exception occurred while scoring the model:\n{}'.format(str(e)))
            return cv

    def _spread_probs_to_all_classes(self, probs, classes_):
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

    def _score(self, cv: GridSearchCV, X_test, y_test) -> Score:
        """
        Scores the performance of a classifier trained in `cv`.
        """
        best_pipeline = cv.best_estimator_
        y_test_pred = best_pipeline.predict_proba(X_test)
        y_test_pred = self._spread_probs_to_all_classes(y_test_pred, best_pipeline.classes_)

        if y_test_pred.shape[1] == 2:
            # binary case
            y_test_pred = y_test_pred[:, 1]

        cross_entropy = log_loss(y_test, y_test_pred)
        gini = self._gini_score(y_test, y_test_pred)

        top_k_acc = self.top_k_acc if self.multi_class else 1
        acc = self._top_k_acc_score(best_pipeline, X_test, y_test)

        try:
            counts = self._get_class_counts_in_nodes(best_pipeline, X_test, y_test)
        except Exception as e:
            self._log_item('ERROR: The following exception occurred while computing node counts:\n{} {}'
                           .format(type(e), e))
            counts = None

        return TreeSurrogate.Score(cv_results=cv.cv_results_,
                                   cv_best_index=cv.best_index_,
                                   tree=cv.best_estimator_.tree_,
                                   cv_n_splits=cv.n_splits_,
                                   cross_entropy=cross_entropy,
                                   gini=gini,
                                   top_k_acc=top_k_acc,
                                   acc=acc,
                                   counts=counts)

    def _get_class_counts_in_nodes(self, best_pipeline, X_test, y_test):
        """
        Returns an array where dimension 0 indexes the nodes of `tree_clf`.
        Dimension 1 indexes the classes known by `tree_clf`.
        Elements tell how many test samples of each class fall into each node.
        """
        X_test = best_pipeline['fsel'].transform(X_test)
        involved_nodes = best_pipeline['clf'].decision_path(X_test)
        if len(involved_nodes.shape) == 1:
            involved_nodes = np.expand_dims(involved_nodes, -1)

        class_counts_per_node = []
        for cls in range(len(self.all_classes)):
            nodes = involved_nodes[y_test == cls]
            counts = np.asarray(np.sum(nodes, axis=0))
            class_counts_per_node.append(counts)

        assert np.all(np.sum(involved_nodes, axis=0) == sum(class_counts_per_node)), \
            'number of samples per node must equal sum_{c in classes}(number of samples per node of c)'

        counts = np.stack(class_counts_per_node, axis=-1)

        assert len(X_test) == np.sum(counts[0]), 'all test samples must pass the root node'

        return counts

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
            self._log_item('Classes {} of the test data were not in the training data of this classifier.'
                           .format(warnings))

        return float(hit_count) / float(len(inputs))

    def _gini_score(self, y_true, y_pred):
        multi_class = 'ovo' if self.multi_class else 'raise'
        auc = roc_auc_score(y_true, y_pred, average='macro',
                            multi_class=multi_class, labels=self.all_class_ids)
        return 2. * auc - 1.


@dataclass
class ConceptCountObservations:
    image_ids: np.ndarray
    concept_counts: np.ndarray
    predicted_classes: np.ndarray

    def __post_init__(self):
        assert len(self.image_ids) == len(self.concept_counts) == len(self.predicted_classes)
        assert not any(i is None for i in self.image_ids)
        assert not np.any(np.isnan(self.concept_counts))
        assert not any(s is None for s in self.predicted_classes)

    def __len__(self):
        return len(self.image_ids)


@dataclass
class TestObservations(ConceptCountObservations):
    pass


@dataclass
class TrainObservations(ConceptCountObservations):

    estimator: str
    perturber: str
    detector: str

    def class_counts(self, all_classes: Iterable[str] = tuple()) -> [Tuple[str, int]]:
        """
        Returns tuples of class names and corresponding counts of observations in the training data.
        """
        counts = Counter(it.chain(self.predicted_classes, all_classes))
        m = 0 if all_classes is None else 1
        return sorted(((s, c - m) for s, c in counts.items()),
                      key=lambda x: (1.0 / (x[1] + 1), x[0]))

    def filter_for_split(self, num_splits=5, security_factor=1.2, all_classes: Iterable[str] = tuple()):
        """
        Filters the training data so that all predicted classes appear at least
        `ceil(security_factor * num_splits)` times.
        Use this to keep only classes where at least one prediction for each CV split is available.
        """
        min_samples = math.ceil(security_factor * float(num_splits))  # need >= num_splits samples for each scene
        enough_samples = list(s for s, c in self.class_counts(all_classes) if c >= min_samples)
        return np.isin(self.predicted_classes, enough_samples)


@dataclass
class FitSurrogatesTask(ComposableDataclass, LoggingMixin):
    """
    Fits TreeSurrogate models on the observations generated by different combinations
    of influence estimator, perturber and detector.
    Returns the grid of resulting model parameters and accuracies.
    """

    @dataclass
    class Results:
        scores: np.ndarray
        all_concept_names: [str]
        influence_estimators: np.ndarray
        perturbers: np.ndarray
        detectors: np.ndarray

    # url of a concept-counts train dataset generated by the module `observe_classifier`
    train_url: str

    # url of perturbations of concept-counts as generated by the module `perturb`
    perturbations_url: str

    # url of a concept-counts test dataset generated by the module `observe_classifier`
    test_url: str

    # how to use spark
    spark_cfg: SparkSessionConfig

    # how to train tree surrogate models
    tree: TreeSurrogate

    # fraction of samples to use from the train dataset
    train_sample_fraction: float = 1.0

    # number of samples to draw from the train dataset for each class.
    # overrides `train_sample_fraction`.
    train_samples_per_class: Optional[float] = None

    # fraction of the training data of each surrogate model
    # for which perturbed data should be added, in the interval (0, 1].
    perturb_fraction: Optional[float] = None

    # seed to use for sampling
    seed: int = np.random.randint(0, np.iinfo(int).max)

    # combinations of (influence estimator, perturber, detector)
    # for which to fit surrogate models.
    # defaults to all available combinations.
    # each string defines a regex filter and can thus match multiple combinations.
    grid: Optional[List[Tuple[str, str, str]]] = None

    def __post_init__(self):
        super().__post_init__()

        assert 0 < self.train_sample_fraction <= 1.0
        assert self.perturb_fraction is None or 0 < self.perturb_fraction <= 1.0

        self.perturbations_fields = {Field.IMAGE_ID, Field.PREDICTED_CLASS, Field.PERTURBED_IMAGE_ID,
                                     Field.INFLUENCE_ESTIMATOR, Field.PERTURBER, Field.DETECTOR}
        self.supervised_fields = {Field.IMAGE_ID, Field.PREDICTED_CLASS}

    @staticmethod
    def _decode(df: DataFrame, all_concept_names: [str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pd_df = df.toPandas()
        return pd_df.loc[:, Field.IMAGE_ID.name].to_numpy(dtype=Field.IMAGE_ID.numpy_dtype), \
            pd_df.loc[:, all_concept_names].to_numpy(dtype=Field.CONCEPT_COUNTS.numpy_dtype), \
            pd_df.loc[:, Field.PREDICTED_CLASS.name].to_numpy(dtype=Field.PREDICTED_CLASS.numpy_dtype)

    @staticmethod
    def _get_sanitized_schema_from_dataset_url(url: str):
        # this hack is necessary to get a unischema whose fields have the correct hash function
        # from an unpickled unischema :(
        unpickled_schema = get_schema_from_dataset_url(url)
        return Unischema(unpickled_schema._name, [UnischemaField(f.name, f.numpy_dtype, f.shape, f.codec, f.nullable)
                                                  for f in unpickled_schema.fields.values()])

    def _load_dataset(self, url: str, required_fields: {UnischemaField}):
        schema = self._get_sanitized_schema_from_dataset_url(url)
        found_fields = set(schema.fields.values())
        assert found_fields >= required_fields
        return self.spark_cfg.session.read.parquet(url), found_fields - required_fields

    def run(self) -> Results:
        with self._log_task('Training surrogate models'):
            train_df, train_concept_fields = self._load_dataset(self.train_url, self.supervised_fields)
            perturbed_df, perturbed_concept_fields = self._load_dataset(self.perturbations_url,
                                                                        self.perturbations_fields)
            test_df, test_concept_fields = self._load_dataset(self.test_url, self.supervised_fields)

            all_concept_fields = train_concept_fields | perturbed_concept_fields | test_concept_fields
            for concept_field in all_concept_fields:
                concept_name = concept_field.name

                if concept_name not in train_df.columns:
                    self._log_item('Adding concept {} to training data'.format(concept_field.name))
                    train_df = train_df.withColumn(concept_name, sf.lit(0))

                if concept_name not in perturbed_df.columns:
                    self._log_item('Adding concept {} to perturbed data'.format(concept_field.name))
                    perturbed_df = perturbed_df.withColumn(concept_name, sf.lit(0))

                if concept_name not in test_df.columns:
                    self._log_item('Adding concept {} to test data'.format(concept_field.name))
                    test_df = test_df.withColumn(concept_name, sf.lit(0))

            all_concept_names = [f.name for f in all_concept_fields]
            self._log_item('Using {} concepts in total (train + perturbations + test data).'
                           .format(len(all_concept_names)))

            test_obs = TestObservations(*self._decode(test_df, all_concept_names))
            test_df.unpersist()
            self._log_item('Test data comprises {} observations.'.format(len(test_obs)))

            scores = []
            influence_estimators = []
            perturbers = []
            detectors = []

            if self.train_samples_per_class is not None:
                fraction_per_class = {row[Field.PREDICTED_CLASS.name]:
                                      np.clip(float(self.train_samples_per_class) / float(row['count']), 0, 1)
                                      for row in train_df.groupBy(Field.PREDICTED_CLASS.name).count().collect()}
                train_df = train_df.sampleBy(Field.PREDICTED_CLASS.name, fraction_per_class, seed=self.seed)
            else:
                train_df = train_df.sample(self.train_sample_fraction, seed=self.seed)

            train_image_count = train_df.count()

            groups = perturbed_df.select(Field.INFLUENCE_ESTIMATOR.name, Field.PERTURBER.name,
                                         Field.DETECTOR.name).distinct().collect()
            if self.grid is not None:
                groups = [(e, p, d) for e, p, d in groups for e_, p_, d_ in self.grid
                          if re.fullmatch(e_, e) and re.fullmatch(p_, p) and re.fullmatch(d_, d)]

            with self._log_task('Selected hyperparameters for fitting:'):
                for group_no, group in enumerate(groups, start=1):
                    with self._log_task('Combination {}'.format(group_no)):
                        for x in group:
                            self._log_item(x)

            for influence_estimator, perturber, detector in it.chain(((None, None, None),), groups):
                with self._log_task('Fitting surrogate model based on:'):
                    self._log_item('Influence estimator: {}'.format(influence_estimator))
                    self._log_item('Perturber: {}'.format(perturber))
                    self._log_item('Detector: {}'.format(detector))
                    self.tree.log_nesting = self.log_nesting

                    if influence_estimator is None:
                        assert {perturber, detector} == {None}
                        group_df = train_df
                    else:
                        assert None not in {perturber, detector}
                        group_df = perturbed_df.filter((perturbed_df.influence_estimator == influence_estimator)
                                                       & (perturbed_df.perturber == perturber)
                                                       & (perturbed_df.detector == detector)) \
                            .select(*('`{}`'.format(f.name) for f in self.supervised_fields),
                                    *('`{}`'.format(concept_name) for concept_name in all_concept_names))

                        # we should not have more images than in train_df, for a fair comparison.
                        # -> drop images if there are more.
                        # further, drop images to meet the desired `self.perturbation_fraction`
                        f = 1 if self.perturb_fraction is None else self.perturb_fraction
                        target_perturb_image_count = int(np.ceil(train_image_count * f))
                        image_ids = group_df.select(Field.IMAGE_ID.name).limit(target_perturb_image_count)
                        perturb_image_count = image_ids.count()
                        if self.perturb_fraction is not None and perturb_image_count < target_perturb_image_count:
                            self._log_item('WARNING: found not enough perturbations to create a training dataset '
                                           'with the specified fraction of {} perturbations.'
                                           '{} perturbations would be needed, but only {} are available.'
                                           .format(self.perturb_fraction, target_perturb_image_count,
                                                   perturb_image_count))
                        group_df = group_df.join(image_ids, on=Field.IMAGE_ID.name, how='leftsemi')

                        # we should also not have less images than in train_df.
                        # -> fill up from train_df until we have equal numbers.
                        fill_count = train_image_count - perturb_image_count
                        assert fill_count >= 0, 'spark dataframe has returned too many rows'
                        fill_up_df = train_df.join(group_df, on=Field.IMAGE_ID.name, how='anti').limit(fill_count)
                        group_df = group_df.unionByName(fill_up_df)

                        assert group_df.select(Field.IMAGE_ID.name).count() == train_image_count, \
                            'could not ensure the same number of images for the baseline ' \
                            'and perturbed datasets. needs investigation.'

                    # note: group_df.count() is non-deterministic due to the sampling in train_df,
                    # except if a seed is set.
                    train_obs = TrainObservations(*self._decode(group_df, all_concept_names),
                                                  influence_estimator, perturber, detector)
                    self._log_item('We have {} candidate observations.'.format(len(train_obs)))

                    indices = train_obs.filter_for_split(self.tree.k_folds)
                    self._log_item('After filtering for CV, {} observations remain.'.format(np.count_nonzero(indices)))
                    score = self.tree.fit_and_score(train_obs.concept_counts[indices],
                                                    train_obs.predicted_classes[indices],
                                                    test_obs.concept_counts, test_obs.predicted_classes)

                    scores.append(score)
                    influence_estimators.append(influence_estimator)
                    perturbers.append(perturber)
                    detectors.append(detector)

                    group_df.unpersist()  # free memory

            return FitSurrogatesTask.Results(scores=np.asarray(scores, dtype=object),
                                             all_concept_names=all_concept_names,
                                             influence_estimators=np.asarray(influence_estimators),
                                             perturbers=np.asarray(perturbers),
                                             detectors=np.asarray(detectors))
