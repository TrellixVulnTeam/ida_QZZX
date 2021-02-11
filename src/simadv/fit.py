import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Union, Any, Dict, Iterable, Tuple

from petastorm.etl.dataset_metadata import get_schema_from_dataset_url
from petastorm.unischema import Unischema
from petastorm.utils import decode_row
from pyspark.sql import DataFrame
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import itertools as it

from simadv.spark import Field, Schema, SparkSessionConfig


@dataclass
class TreeSurrogate:

    @total_ordering
    @dataclass
    class Score:
        cv: GridSearchCV
        cross_entropy: float
        gini: float
        top_k_acc: int
        acc: float
        counts: np.ndarray

        def __str__(self):
            res = self.cv.cv_results_
            entropies = np.asarray([res['split{}_test_score'.format(i)][self.cv.best_index_]
                                   for i in range(self.cv.n_splits_)])

            s = 'Best model has\n' \
                '-> Mean Training Cross-entropy: {}\n' \
                '-> Training StdDev: {}\n' \
                '-> Params:\n' \
                .format(np.mean(entropies), np.std(entropies))

            for k, v in self.cv.best_params_.items():
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
            logging.error('The following exception occurred while scoring the model:\n{}'.format(str(e)))
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
            logging.error('The following exception occurred while computing node counts:\n{}'.format(str(e)))
            counts = None

        return TreeSurrogate.Score(cv, cross_entropy, gini, top_k_acc, acc, counts)

    @staticmethod
    def _get_class_counts_in_nodes(best_pipeline, X_test, y_test):
        """
        Returns an array where dimension 0 indexes the nodes of `tree_clf`.
        Dimension 1 indexes the classes known by `tree_clf`.
        Elements tell how many test samples of each class fall into each node.
        """
        X_test = best_pipeline['fsel'].transform(X_test)
        involved_nodes = best_pipeline['clf'].decision_path(X_test)

        class_counts_per_node = []
        for cls in best_pipeline['clf'].classes_:
            nodes = involved_nodes[y_test == cls]
            counts = np.asarray(np.sum(nodes, axis=0)).squeeze()
            class_counts_per_node.append(counts)

        assert np.all(np.sum(involved_nodes, axis=0) == sum(class_counts_per_node))

        counts = np.stack(class_counts_per_node, axis=-1)

        assert len(X_test) == np.sum(counts[0])  # all test samples must pass the root node

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
            logging.warning('Classes {} were not in the training subset '
                            'of this classifier!'.format(warnings))

        return float(hit_count) / float(len(inputs))

    def _gini_score(self, y_true, y_pred):
        multi_class = 'ovo' if self.multi_class else 'raise'
        auc = roc_auc_score(y_true, y_pred, average='macro',
                            multi_class=multi_class, labels=self.all_class_ids)
        return 2. * auc - 1.


@dataclass
class ObjectCountObservations:
    image_ids: np.ndarray
    object_counts: np.ndarray
    predicted_classes: np.ndarray

    def __post_init__(self):
        assert len(self.image_ids) == len(self.object_counts) == len(self.predicted_classes)
        assert not any(i is None for i in self.image_ids)
        assert not np.any(np.isnan(self.object_counts))
        assert not any(s is None for s in self.predicted_classes)

    def __len__(self):
        return len(self.image_ids)


@dataclass
class TestObservations(ObjectCountObservations):
    pass


@dataclass
class TrainObservations(ObjectCountObservations):

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
        Takes in a boolean array `subset` that specifies a subset of the training data.
        Filters the subset so that all predicted classes appear at least `ceil(security_factor * num_splits)` times.
        Use this to keep only classes where at least one prediction for each CV split is available.
        """
        min_samples = math.ceil(security_factor * float(num_splits))  # need >= num_splits samples for each scene
        enough_samples = list(s for s, c in self.class_counts(all_classes) if c >= min_samples)
        return np.isin(self.predicted_classes, enough_samples)


@dataclass
class FitSurrogatesTask:
    """
    Fits TreeSurrogate models on the observations generated by different combinations
    of influence estimator, perturber and detector, as stored in a `Schema.PERTURBED_OBJECT_COUNTS` input schema.
    Returns the grid of resulting model parameters and accuracies.
    """
    
    @dataclass
    class Results:
        scores: np.ndarray
        influence_estimators: np.ndarray
        perturbers: np.ndarray
        detectors: np.ndarray
    
    train_input_url: str
    train_schema: Unischema = field(default=Schema.PERTURBED_CONCEPT_COUNTS, init=False)

    test_input_url: str
    test_schema: Unischema = field(default=Schema.TEST, init=False)

    spark_cfg: SparkSessionConfig
    tree: TreeSurrogate

    @staticmethod
    def _decode(df: DataFrame, schema) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_ids = []
        object_counts = []
        predicted_classes = []

        for df_row in df.toPandas().to_dict('records'):
            row = decode_row(df_row, schema)
            image_ids.append(row[Field.IMAGE_ID.name])
            object_counts.append(row[Field.CONCEPT_COUNTS.name])
            predicted_classes.append(row[Field.PREDICTED_CLASS.name])

        return np.asarray(image_ids), np.asarray(object_counts), np.asarray(predicted_classes)

    def run(self) -> Results:
        train_schema = get_schema_from_dataset_url(self.train_input_url)
        assert set(train_schema.fields) >= set(Schema.PERTURBED_CONCEPT_COUNTS.fields)
        train_df = self.spark_cfg.session.read.parquet(self.train_input_url)

        test_schema = get_schema_from_dataset_url(self.test_input_url)
        assert set(test_schema.fields) >= set(Schema.TEST.fields)
        test_df = self.spark_cfg.session.read.parquet(self.test_input_url)
        test_obs = TestObservations(*self._decode(test_df, test_schema))
        logging.info('Test data comprises {} observations.'.format(len(test_obs)))

        scores = []
        influence_estimators = []
        perturbers = []
        detectors = []

        groups = train_df.select(Field.INFLUENCE_ESTIMATOR.name, Field.PERTURBER.name,
                                 Field.DETECTOR.name).distinct()
        
        for influence_estimator, perturber, detector in groups.collect():
            logging.info('Fitting for:\n{}\n{}\n{}'.format(influence_estimator, perturber, detector))
            group_key = {influence_estimator, perturber, detector}
            if None in group_key:
                if group_key != {None}:
                    raise ValueError('Influence_estimator, perturber and detector must either all three be None'
                                     'or all three must be a string.')
                group_df = train_df.filter(train_df.influence_estimator.isNull() & train_df.perturber.isNull()
                                           & train_df.detector.isNull())
            else:
                group_df = train_df.filter((train_df.influence_estimator == influence_estimator)
                                           & (train_df.perturber == perturber)
                                           & (train_df.detector == detector))
            logging.info('We have {} candidate observations.'.format(group_df.count()))
            train_obs = TrainObservations(*self._decode(group_df, train_schema),
                                          influence_estimator, perturber, detector)

            indices = train_obs.filter_for_split(self.tree.k_folds)
            logging.info('After filtering for CV, {} observations remain.'.format(np.count_nonzero(indices)))
            score = self.tree.fit_and_score(train_obs.object_counts[indices], train_obs.predicted_classes[indices],
                                            test_obs.object_counts, test_obs.predicted_classes)

            scores.append(score)
            influence_estimators.append(influence_estimator)
            perturbers.append(perturber)
            detectors.append(detector)
        
        return FitSurrogatesTask.Results(scores=np.asarray(scores, dtype=object),
                                         influence_estimators=np.asarray(influence_estimators),
                                         perturbers=np.asarray(perturbers),
                                         detectors=np.asarray(detectors))
