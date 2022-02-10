import abc
from typing import Optional, Any, Tuple, Iterable, Dict, Type, Mapping

import numpy as np
import torch
from captum.attr import Attribution

from ida.interpret.common import Interpreter
from ida.torch_extensions.classifier import TorchImageClassifier


class Type2Explainer(abc.ABC):

    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter):
        self.classifier = classifier
        self.interpreter = interpreter

    @property
    def stats(self) -> Dict[str, Any]:
        return {}

    def calibrate(self, images_iter: Iterable[Tuple[str, np.ndarray]]):
        pass

    @abc.abstractmethod
    def __call__(self, image: np.ndarray, image_id: Optional[str] = None) \
            -> Iterable[Tuple[int, np.ndarray, bool]]:
        """
        Takes an input *image* of the classifier and returns the location and influence of each instance of
        an interpretable concept on *image* on the output of the classifier.
        The influence estimator can use an additional *image_id* to lookup interpretable concepts of *image*.
        """
        pass


class NoType2Explainer(Type2Explainer):

    def __call__(self, image: np.ndarray, image_id: Optional[str] = None, **kwargs) \
            -> Iterable[Tuple[int, np.ndarray, bool]]:
        """
        Marks all interpretable concepts on *image* / the image represented by *image_id* as influential.
        """
        for concept_id, mask in self.interpreter(image, image_id):
            yield concept_id, mask, True

    def __str__(self):
        return 'NoType2Explainer'


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
