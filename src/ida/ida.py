import abc
import itertools as it
import time
from collections import Counter
from typing import Iterable, Tuple, Dict, Any, Optional, List, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from ida.interpret.common import Interpreter
from ida.type1.common import Type1Explainer
from ida.type2.common import Type2Explainer
from ida.common import NestedLogger


def class_counts(predicted_classes, all_classes) -> [Tuple[str, int]]:
    """
    Returns tuples of class ids and corresponding counts of observations in the training data.
    """
    counts = Counter(it.chain(predicted_classes, all_classes))
    m = 0 if all_classes is None else 1
    return sorted(((s, c - m) for s, c in counts.items()),
                  key=lambda x: (1.0 / (x[1] + 1), x[0]))


def filter_for_split(predicted_classes, min_k_folds, all_classes):
    """
    Filters the training data so that all predicted classes appear at least
    `ceil(security_factor * num_splits)` times.
    Use this to keep only classes where at least one prediction for each CV split is available.
    """
    enough_samples = list(s for s, c in class_counts(predicted_classes, all_classes) if c >= min_k_folds)
    return np.isin(predicted_classes, enough_samples)


def determine_k_folds(predicted_classes: [int], max_k_folds: int):
    return min(max_k_folds, Counter(predicted_classes).most_common()[-1][1])


def run_cv(inputs: List[Any],
           predicted_classes: List[int],
           all_classes: Optional[List[int]],
           pipeline: Pipeline,
           param_grid: Dict[str, Any],
           min_k_folds: int = 5,
           max_k_folds: int = 5,
           n_jobs: int = 62,
           pre_dispatch: Union[int, str] = 'n_jobs',
           scoring: str = 'roc_auc_ovo') -> GridSearchCV:
    inputs = np.asarray(inputs, dtype=object)
    predicted_classes = np.asarray(predicted_classes)

    assert min_k_folds <= max_k_folds
    indices = filter_for_split(predicted_classes=predicted_classes,
                               min_k_folds=min_k_folds,
                               all_classes=all_classes)
    if not np.any(indices):
        indices = np.ones_like(indices, dtype=bool)
        cv = KFold(n_splits=min_k_folds)
    else:
        k_folds = determine_k_folds(predicted_classes[indices], max_k_folds)
        cv = StratifiedKFold(n_splits=k_folds)

    search = GridSearchCV(estimator=pipeline,
                          param_grid=param_grid,
                          cv=cv,
                          n_jobs=n_jobs,
                          pre_dispatch=pre_dispatch,
                          scoring=scoring)
    return search.fit(inputs[indices], predicted_classes[indices])


class InterpretPickTransformer(abc.ABC, BaseEstimator, TransformerMixin, NestedLogger):

    def __init__(self,
                 type2: Type2Explainer,
                 threshold: float = .5,
                 quantile: bool = False,
                 random_state: int = 42,
                 max_num_type2_calls: Optional[int] = None):
        self.type2 = type2
        self.threshold = threshold
        self.quantile = quantile
        self.log_frequency_s = 5.
        self.random_state = random_state
        self.max_num_type2_calls = max_num_type2_calls

    @property
    def interpreter(self) -> Interpreter:
        return self.type2.interpreter

    def _sample(self, rng, X):
        if self.max_num_type2_calls is None:
            return X
        if len(X) <= self.max_num_type2_calls:
            return X
        return rng.choice(X, self.max_num_type2_calls, replace=False)

    def _get_counts(self, X):
        concept_counts = []
        influential_concept_counts = []

        rng = np.random.default_rng(self.random_state)
        last_log_time = time.time()
        for obs_no, (image_id, image) in enumerate(self._sample(rng, X)):
            concept_ids, _, influences = list(zip(*self.type2(image=image, image_id=image_id)))
            counts = self.interpreter.concept_ids_to_counts(concept_ids)
            concept_counts.append(counts)
            influential_concept_ids = [c for c, i in zip(concept_ids, influences) if i]
            influential_counts = self.interpreter.concept_ids_to_counts(influential_concept_ids)
            influential_concept_counts.append(influential_counts)

            current_time = time.time()
            if current_time - last_log_time > self.log_frequency_s:
                with self.log_task('Status update'):
                    total_count = obs_no + 1
                    self.log_item(f'Processed {total_count} observations')
                last_log_time = current_time
        return concept_counts, influential_concept_counts

    def _evaluate_counters(self, counts: [[int]], influential_counts: [[int]]):
        counts = np.sum(np.asarray(counts), axis=0)
        influential_counts = np.sum(np.asarray(influential_counts), axis=0)

        self.num_influential_concept_instances_ = np.sum(influential_counts)
        self.stats_ = self.type2.stats

        concept_influences = []
        for concept_id in range(len(self.interpreter.concepts)):
            influential_count = float(influential_counts[concept_id])
            total_count = float(counts[concept_id])
            if total_count == 0.:
                concept_influences.append(0.)
            else:
                concept_influences.append(influential_count / total_count)

        self.concept_influences_ = concept_influences
        inf_by_name = sorted(zip(self.interpreter.concepts, self.concept_influences_),
                             key=lambda x: x[1],
                             reverse=True)
        self.log_item(f'Final concept influences: {inf_by_name}')
        self.picked_concepts_ = self.get_influential_concepts()
        self.picked_concept_names_ = np.asarray(self.interpreter.concepts)[self.picked_concepts_]
        self.log_item(f'Picked concepts: {self.picked_concept_names_}')

    def fit(self, X, y=None):
        with self.log_task('Observing concept influences...'):
            counts, influential_counts = self._get_counts(X)
            self._evaluate_counters(counts, influential_counts)

    def transform(self, X):
        with self.log_task('Interpreting inputs...'):
            counts, influential_counts = self._get_counts(X)
            return np.asarray(counts)[:, self.get_influential_concepts()]

    def fit_transform(self, X, y=None, **fit_params):
        with self.log_task('Observing concept influences...'):
            counts, influential_counts = self._get_counts(X)
            self._evaluate_counters(counts, influential_counts)
        return np.asarray(counts)[:, self.get_influential_concepts()]

    def get_influential_concepts(self) -> [int]:
        if self.quantile:
            if len(np.unique(self.concept_influences_)) == 1:
                # treat all concepts as influential
                return list(range(len(self.concept_influences_)))

            threshold = np.quantile(self.concept_influences_, self.threshold)
        else:
            threshold = self.threshold
        return [idx for idx, i in enumerate(self.concept_influences_) if i > threshold]


def ipa(image_iter: Iterable[Tuple[str, np.ndarray]],
        type2: Type2Explainer,
        type1: Type1Explainer,
        param_grid: Dict[str, Any],
        cv_params: Dict[str, Any],
        random_state: int,
        log_nesting: int = 0) -> GridSearchCV:

    logger = NestedLogger()
    logger.log_nesting = log_nesting
    with logger.log_task('Observing classifier...'):
        inputs = []
        predicted_classes = []
        for obs_no, (image_id, image) in enumerate(image_iter):
            inputs.append((image_id, image))
            predicted_classes.append(type2.classifier.predict_single(image=image, image_id=image_id))

    with logger.log_task('Running IPA...'):
        pipeline = Pipeline([
            ('interpret-pick', InterpretPickTransformer(type2, random_state=random_state)),
            ('approximate', type1.create_pipeline(random_state=random_state))
        ])
        cv = run_cv(inputs=inputs,
                    predicted_classes=predicted_classes,
                    all_classes=list(range(type2.classifier.num_classes)),
                    pipeline=pipeline,
                    param_grid=param_grid,
                    **cv_params)

        return cv
