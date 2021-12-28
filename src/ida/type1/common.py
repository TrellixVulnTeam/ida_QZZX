import abc
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.pipeline import Pipeline

from ida.interpret.common import Interpreter
from ida.torch_extensions.classifier import TorchImageClassifier


class Type1Explainer(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def create_pipeline(random_state: int) -> Pipeline:
        pass

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
    def serialize(pipeline: Pipeline) -> Dict[str, Any]:
        pass

    @staticmethod
    @abc.abstractmethod
    def plot(pipeline: Pipeline, **kwargs):
        pass


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
                                          picked_concepts: [int],
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
    counts = np.asarray(counts)[:, picked_concepts]
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
            cf_counts = np.asarray(cf_counts)[picked_concepts]
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
            f'coverage_of_top_{k}_influence': num_covered_influential_cfs / num_influential_cfs,
            'coverage_of_counterfactuals': num_correctly_predicted_influential_cfs / num_influential_cfs}
