from typing import Type, Optional, Tuple, Iterable

import numpy as np
from captum.attr import GradientAttribution, Saliency, IntegratedGradients, DeepLift

from liga.interpret.common import Interpreter
from liga.type2.common import Type2Explainer, CaptumAttributionWrapper
from liga.torch_extensions.classifier import TorchImageClassifier


class GradientAttributionType2Explainer(Type2Explainer):

    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter,
                 gradient_attribution_method: Type[GradientAttribution],
                 takes_additional_attribution_args: bool = False,
                 **additional_init_args):
        super().__init__(classifier=classifier,
                         interpreter=interpreter)
        self.captum_wrapper = CaptumAttributionWrapper(classifier=classifier,
                                                       attribution_method=gradient_attribution_method,
                                                       **additional_init_args)
        self.takes_additional_attribution_args = takes_additional_attribution_args

    def __call__(self, image: np.ndarray, image_id: Optional[str] = None, **kwargs) -> Iterable[Tuple[int, float]]:
        feature_influences = None

        for concept_id, mask in self.interpreter(image=image,
                                                 image_id=image_id,
                                                 **kwargs):
            if not self.takes_additional_attribution_args:
                kwargs = {}

            if feature_influences is None:
                feature_influences = (self.captum_wrapper(image=image,
                                                          additional_attribution_args=kwargs)
                                      .transpose(1, 2, 0))  # CxHxW -> HxWxC
            yield concept_id, np.sum(feature_influences[mask]).item()


class SaliencyType2Explainer(GradientAttributionType2Explainer):
    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter):
        super().__init__(classifier=classifier,
                         interpreter=interpreter,
                         gradient_attribution_method=Saliency,
                         takes_additional_attribution_args=True)

    def __call__(self, image: np.ndarray, image_id: Optional[str] = None, **kwargs) -> [float]:
        return super().__call__(image=image,
                                image_id=image_id,
                                abs=False)  # we want positive *and* negative attribution values
        # we ignore kwargs here because the Saliency class cannot deal with them

    def __str__(self):
        return 'saliency'


class IntegratedGradientsType2Explainer(GradientAttributionType2Explainer):
    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter):
        super().__init__(classifier=classifier,
                         interpreter=interpreter,
                         gradient_attribution_method=IntegratedGradients)

    def __str__(self):
        return 'igrad'


class DeepLiftType2Explainer(GradientAttributionType2Explainer):
    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter):
        super().__init__(classifier=classifier,
                         interpreter=interpreter,
                         gradient_attribution_method=DeepLift)

    def __str__(self):
        return 'deeplift'
