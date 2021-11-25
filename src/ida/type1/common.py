import abc
from collections import Counter
from functools import partial
from typing import List, Dict, Any, Optional, Union, Tuple

import itertools as it
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ida.common import NestedLogger
from ida.interpret.common import Interpreter
from ida.torch_extensions.classifier import TorchImageClassifier


class Type1Explainer(abc.ABC):

    @abc.abstractmethod
    def __call__(self,
                 concept_counts: List[List[int]],
                 predicted_classes: List[int],
                 all_classes: Optional[List[int]],
                 **kwargs) -> Pipeline:
        """
        Trains a scikit-learn model to predict *predicted_classes* based on *concept_counts*.
        """

    @staticmethod
    @abc.abstractmethod
    def get_complexity_metrics(pipeline: Pipeline) -> Dict[str, Any]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_fitted_params(pipeline: Pipeline) -> Dict[str, Any]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_plot_representation(pipeline: Pipeline) -> Dict[str, str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def plot(experiment_name: str, exp_no: int, rep_no: int, **kwargs):
        pass


class CrossValidatedType1Explainer(Type1Explainer, abc.ABC):

    def __call__(self,
                 concept_counts: List[List[int]],
                 predicted_classes: List[int],
                 all_classes: Optional[List[int]],
                 random_state: int = 42,
                 min_k_folds: int = 5,
                 max_k_folds: int = 5,
                 top_k_acc: int = 5,
                 n_jobs: int = 10,
                 pre_dispatch: Union[int, str] = 'n_jobs',
                 select_top_k_influential_concepts: int = 30,
                 scoring: Optional[str] = None,
                 log_nesting: int = 0,
                 **fit_params) -> Pipeline:
        if scoring is None:
            logger = NestedLogger()
            logger.log_nesting = log_nesting
            scoring = partial(top_k_accuracy_score,
                              k=top_k_acc,
                              logger=logger,
                              all_classes=all_classes)

        concept_counts = np.asarray(concept_counts)
        predicted_classes = np.asarray(predicted_classes)

        assert min_k_folds <= max_k_folds
        indices = self.filter_for_split(predicted_classes=predicted_classes,
                                        min_k_folds=min_k_folds,
                                        all_classes=all_classes)
        if not np.any(indices):
            indices = np.ones_like(indices, dtype=bool)
            cv = KFold(n_splits=min_k_folds)
        else:
            k_folds = self.determine_k_folds(predicted_classes[indices], max_k_folds)
            cv = StratifiedKFold(n_splits=k_folds)

        search = GridSearchCV(estimator=self.get_pipeline(random_state=random_state),
                              param_grid=fit_params,
                              cv=cv,
                              n_jobs=n_jobs,
                              pre_dispatch=pre_dispatch,
                              scoring=scoring)
        search.fit(concept_counts[indices], predicted_classes[indices])
        return search.best_estimator_

    @abc.abstractmethod
    def get_pipeline(self, random_state: int) -> Pipeline:
        pass

    @staticmethod
    def get_fitted_params(pipeline: Pipeline) -> Dict[str, Any]:
        return {**pipeline.get_params()}

    @staticmethod
    def class_counts(predicted_classes, all_classes) -> [Tuple[str, int]]:
        """
        Returns tuples of class ids and corresponding counts of observations in the training data.
        """
        counts = Counter(it.chain(predicted_classes, all_classes))
        m = 0 if all_classes is None else 1
        return sorted(((s, c - m) for s, c in counts.items()),
                      key=lambda x: (1.0 / (x[1] + 1), x[0]))

    def filter_for_split(self, predicted_classes, min_k_folds, all_classes):
        """
        Filters the training data so that all predicted classes appear at least
        `ceil(security_factor * num_splits)` times.
        Use this to keep only classes where at least one prediction for each CV split is available.
        """
        enough_samples = list(s for s, c in self.class_counts(predicted_classes, all_classes) if c >= min_k_folds)
        return np.isin(predicted_classes, enough_samples)

    @staticmethod
    def determine_k_folds(predicted_classes: [int], max_k_folds: int):
        return min(max_k_folds, Counter(predicted_classes).most_common()[-1][1])


def _get_indices_with_top_k_match(surrogate: Pipeline,
                                  counts: List[List[int]],
                                  target_classes: List[int],
                                  k: int,
                                  logger=None,
                                  all_classes=None):
    class_id_mapping = {cls: cls_id for cls_id, cls in enumerate(surrogate.classes_)}
    predicted = surrogate.predict_proba(counts)
    top_k_class_ids = np.argsort(predicted)[:, -k:]

    target_outputs_translated = np.asarray([class_id_mapping.get(y_true, np.nan) for y_true in target_classes])
    warnings = np.asarray(target_classes)[np.isnan(target_outputs_translated)]
    if len(warnings) > 0 and logger is not None:
        logger.log_item('Classes {} of the test data were not in the training data of this classifier.'
                        .format(sorted([all_classes[c] for c in warnings])))

    return np.isin(target_outputs_translated, top_k_class_ids)


def top_k_accuracy_score(estimator,
                         inputs,
                         target_outputs,
                         k=5,
                         logger=None,
                         all_classes=None):
    idx = _get_indices_with_top_k_match(surrogate=estimator,
                                        counts=inputs,
                                        target_classes=target_outputs,
                                        k=k,
                                        logger=logger,
                                        all_classes=all_classes)

    return float(np.count_nonzero(idx)) / float(idx.shape[0])


def counterfactual_top_k_accuracy_score(surrogate: Pipeline,
                                        images: List[Tuple[str, np.ndarray]],
                                        counts: List[List[int]],
                                        target_classes: List[int],
                                        classifier: TorchImageClassifier,
                                        interpreter: Interpreter,
                                        max_perturbed_area: float,
                                        max_concept_overlap: float,
                                        k=5,
                                        logger=None,
                                        all_classes=None):
    """
    This score is different from "normal" top-k accuracy, because it requires that predictions
    are correct for a pair of an input and its "counterfactual twin" where a concept has been removed.
    """
    idx = _get_indices_with_top_k_match(surrogate=surrogate,
                                        counts=counts,
                                        target_classes=target_classes,
                                        k=k,
                                        logger=logger,
                                        all_classes=all_classes)
    pair_count = 0.
    faithful_count = 0.
    for has_top_k_match, (image_id, image) in zip(idx, images):
        cf_iter = interpreter.get_counterfactuals(image=image,
                                                  image_id=image_id,
                                                  max_perturbed_area=max_perturbed_area,
                                                  max_concept_overlap=max_concept_overlap)
        for cf_counts, cf_image in cf_iter:
            pair_count += 1
            if has_top_k_match:
                cf_true_class = classifier.predict_single(image=cf_image)
                cf_idx = _get_indices_with_top_k_match(surrogate=surrogate,
                                                       counts=[cf_counts],
                                                       target_classes=[cf_true_class],
                                                       k=k)
                if cf_idx[0]:
                    faithful_count += 1.

    return float(faithful_count) / float(pair_count)
