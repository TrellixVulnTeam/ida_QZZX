import abc
import functools
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, Tuple

import numpy as np

RowDict = Dict[str, Any]


class ComposableDataclass:
    """
    Base class for dataclasses that one can use with multiple inheritance.
    """

    def __post_init__(self):
        # just intercept the __post_init__ calls so they aren't relayed to `object`
        # see https://stackoverflow.com/questions/59986413/
        pass


@dataclass
class LoggingConfig:
    level: str = 'info'  # the logging level

    def __post_init__(self):
        logging.basicConfig(level=str.upper(self.level))


@dataclass
class NestedLogger:
    log_nesting: int = field(default=0, init=False)

    def log(self, msg):
        prefix = '--' * self.log_nesting + ' ' if self.log_nesting else ''
        logging.info(prefix + msg)

    def log_item(self, msg):
        self.log('<{}/>'.format(msg))

    def _log_group_start(self, msg):
        self.log('<{}>'.format(msg))
        self.log_nesting += 1

    def _log_group_end(self):
        self.log_nesting -= 1
        self.log('<done/>')

    @contextmanager
    def log_task(self, msg):
        self._log_group_start(msg)
        try:
            yield None
        finally:
            self._log_group_end()


@dataclass
class ClassificationTask:
    name: str

    @property
    @abc.abstractmethod
    def class_names(self) -> [str]:
        pass


class Classifier(abc.ABC):
    """
    Abstract base class for classifiers.
    """

    def __init__(self,
                 name: str,
                 num_classes: int,
                 memoize_predictions: bool = False):
        """
        :param name: name of this model
        :param num_classes: how many classes this classifier discriminates
        :param memoize_predictions: how many predictions to cache for subsequent calls
        """
        self.name = name
        self.num_classes = num_classes
        self.memoize_predictions = memoize_predictions
        self.prediction_cache = {}

    @abc.abstractmethod
    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts probability distributions NxC over the C classes for the given N inputs.
        The distributions are encoded as a float array.
        """

    def predict_single(self, image: np.ndarray, image_id: Optional[str] = None) -> int:
        if image_id in self.prediction_cache.items():
            return self.prediction_cache[image_id]

        pred = np.argmax(self.predict_proba(np.expand_dims(image, 0))[0]).item()
        if self.memoize_predictions:
            self.prediction_cache[image_id] = pred
        return pred


class ImageIdProvider(abc.ABC):

    @abc.abstractmethod
    def get_image_id(self, image_path: Path, subset: Optional[str] = None) -> str:
        """
        Returns the image id of the image located at `image_path`.
        In some datasets, the image id depends on the `subset`, i.e. validation or train subset.
        In this case, the subset must be specified.
        """


class ImageClassProvider(abc.ABC):

    @abc.abstractmethod
    def get_image_class(self, image_id: str, subset: Optional[str] = None) -> int:
        """
        Returns the ground-truth class for the image identified by `image_id`.
        In some datasets image ids are only unique per `subset`, i.e. validation or train subset.
        In this case, the subset must be specified.
        Raises a `KeyError` if no class is known for `image_id`.
        """


class ImageObjectProvider(abc.ABC):

    @abc.abstractmethod
    def object_names(self) -> [str]:
        """
        Returns a list of all object names that this class can provide for images.
        The index of an object name in this list represents the id of the object.
        """

    @abc.abstractmethod
    def get_object_bounding_boxes(self, image_id: str, subset: Optional[str] = None) \
            -> Iterable[Tuple[int, int, int, int, int]]:
        """
        Returns the ground-truth bounding boxes for the image identified by `image_id`.
        Each box is a 5-tuple `(object_id, x_min, y_min, x_max, y_max)`.
        `object_id` identifies an object in the list of names provided by the method `object_names()`.
        In some datasets image ids are only unique per `subset`, i.e. validation or train subset.
        In this case, the subset must be specified.
        Raises a `KeyError` if no bounding box is known for `image_id`.
        """

    def get_implied_objects(self, object_id: int) -> Iterable[int]:
        """
        Subclasses can override this if some concepts imply others.
        """
        return []
