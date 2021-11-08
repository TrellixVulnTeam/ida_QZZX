import abc
from typing import List, TypeVar, Generic, Dict, Any

from sklearn.base import ClassifierMixin

T = TypeVar('T', bound=ClassifierMixin)


class Type1Explainer(abc.ABC, Generic[T]):

    @abc.abstractmethod
    def __call__(self,
                 concept_counts: List[List[int]],
                 predicted_classes: List[int],
                 **kwargs) -> T:
        """
        Trains a scikit-learn model to predict *classes* based on *concept_counts*.
        """

    @abc.abstractmethod
    def get_complexity_metrics(self, model: T) -> Dict[str, Any]:
        pass
