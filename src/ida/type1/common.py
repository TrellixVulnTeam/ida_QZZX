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
                 **fit_params) -> Pipeline:
        if scoring is None:
            scoring = partial(top_k_accuracy_score, k=top_k_acc)

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


def _get_top_k_classes(surrogate: Pipeline,
                       counts: List[List[int]],
                       k: int):
    predicted = surrogate.predict_proba(counts)
    top_k_class_ids = np.argsort(predicted)[:, -k:]

    def translate(class_idx: int):
        return surrogate.classes_[class_idx]

    return translate(top_k_class_ids)


def _get_indices_with_match(target_classes: np.ndarray,
                            top_k_predicted_classes: np.ndarray):
    return np.any(target_classes[..., None] == top_k_predicted_classes, axis=1)


def top_k_accuracy_score(surrogate: Pipeline,
                         counts: List[List[int]],
                         target_classes: List[int],
                         k=5):
    top_k_class_ids = _get_top_k_classes(surrogate=surrogate,
                                         counts=counts,
                                         k=k)
    idx = _get_indices_with_match(target_classes=np.asarray(target_classes, dtype=int),
                                  top_k_predicted_classes=top_k_class_ids)
    return float(np.count_nonzero(idx)) / float(idx.shape[0])


def counterfactual_top_k_accuracy_metrics(surrogate: Pipeline,
                                          images: List[Tuple[str, np.ndarray]],
                                          counts: List[List[int]],
                                          target_classes: List[int],
                                          classifier: TorchImageClassifier,
                                          interpreter: Interpreter,
                                          max_perturbed_area: float,
                                          max_concept_overlap: float,
                                          k=5) -> Dict[str, Any]:
    """
    This score is different from "normal" top-k accuracy, because it requires that predictions
    are correct for a pair of an input and its "counterfactual twin" where a concept has been removed.
    """
    top_k_predicted_class_ids = _get_top_k_classes(surrogate=surrogate,
                                                   counts=counts,
                                                   k=k)
    num_inputs = 0.
    num_cfs = 0.
    num_sensitive_inputs = 0.
    num_influential_cfs = 0.
    num_covered_influential_cfs = 0.
    num_correctly_predicted_influential_cfs = 0.

    for predicted_class_ids, true_class, (image_id, image) in zip(top_k_predicted_class_ids,
                                                                  target_classes,
                                                                  images):
        num_inputs += 1.
        cf_iter = interpreter.get_counterfactuals(image=image,
                                                  image_id=image_id,
                                                  max_perturbed_area=max_perturbed_area,
                                                  max_concept_overlap=max_concept_overlap)

        found_change = False

        for cf_counts, cf_image in cf_iter:
            num_cfs += 1.
            cf_true_class = classifier.predict_single(image=cf_image)
            if cf_true_class != true_class:
                found_change = True
                num_influential_cfs += 1.
                cf_predicted_class_ids = _get_top_k_classes(surrogate=surrogate,
                                                            counts=[cf_counts],
                                                            k=k)[0]

                if np.any(predicted_class_ids != cf_predicted_class_ids):
                    num_covered_influential_cfs += 1.

                if true_class == predicted_class_ids[0] and cf_true_class == cf_predicted_class_ids[0]:
                    num_correctly_predicted_influential_cfs += 1.

        if found_change:
            num_sensitive_inputs += 1.

    return {'fraction_of_sensitive_inputs': num_sensitive_inputs / num_inputs,
            'fraction_of_influential_concepts': num_influential_cfs / num_cfs,
            'coverage_of_top_{}_influence'.format(k): num_covered_influential_cfs / num_influential_cfs,
            'coverage_of_counterfactuals'.format(k): num_correctly_predicted_influential_cfs / num_influential_cfs}
