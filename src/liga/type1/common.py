import abc
from typing import List, TypeVar, Generic, Dict, Any, Optional

from sklearn.base import ClassifierMixin

T = TypeVar('T', bound=ClassifierMixin)


class Type1Explainer(abc.ABC, Generic[T]):

    @abc.abstractmethod
    def __call__(self,
                 concept_counts: List[List[int]],
                 predicted_classes: List[int],
                 all_classes: Optional[List[int]],
                 **kwargs) -> T:
        """
        Trains a scikit-learn model to predict *predicted_classes* based on *concept_counts*.
        Uses *concept_influence_order* to select the most influential concepts.
        """

    @staticmethod
    @abc.abstractmethod
    def get_complexity_metrics(model: T) -> Dict[str, Any]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_fitted_params(model: T) -> Dict[str, Any]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_plot_representation(model: T) -> Dict[str, str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def plot(experiment_name: str, exp_no: int, rep_no: int, **kwargs):
        pass
