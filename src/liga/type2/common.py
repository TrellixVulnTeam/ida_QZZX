import abc
import itertools as it
from typing import Optional, Type, Mapping, Any, Tuple, Iterable

import numpy as np
import torch
from captum.attr import Attribution

from liga.interpret.common import Interpreter
from liga.torch_extensions.classifier import TorchImageClassifier


class Type2Explainer(abc.ABC):

    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter):
        self.classifier = classifier
        self.interpreter = interpreter

    @abc.abstractmethod
    def __call__(self, image: np.ndarray, image_id: Optional[str] = None, **kwargs) \
            -> Iterable[Tuple[int, np.ndarray, float]]:
        """
        Takes an input *image* of the classifier and returns the location and influence of each instance of
        an interpretable concept on *image* on the output of the classifier.
        The influence estimator can use an additional *image_id* to lookup interpretable concepts of *image*.
        """
        pass


class CaptumAttributionWrapper:

    def __init__(self,
                 classifier: TorchImageClassifier,
                 attribution_method: Type[Attribution],
                 **additional_init_args):
        self.classifier = classifier
        self.attr_method = attribution_method(self.classifier.torch_model,
                                              **additional_init_args)

    def __call__(self,
                 image: np.ndarray,
                 additional_attribution_args: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        if additional_attribution_args is None:
            additional_attribution_args = {}
        predicted_class = self.classifier.predict_single(image)
        image_tensor = (torch.from_numpy(np.expand_dims(image, 0))
                        .to(self.classifier.device,
                            dtype=torch.float)
                        .permute(0, 3, 1, 2)
                        .contiguous())  # H x W x C  -->  C x H x W
        attr = (self.attr_method.attribute(image_tensor,
                                           target=int(predicted_class),
                                           **additional_attribution_args)
                .squeeze(0)
                .cpu()
                .detach()
                .numpy()
                .astype(np.float_))
        return attr
