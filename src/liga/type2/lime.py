from functools import reduce
from typing import Optional, Callable, Tuple, List

import numpy as np
import torch
from captum._utils.models import SkLearnLasso, Model
from captum.attr import LimeBase
from captum.attr._core.lime import get_exp_kernel_similarity_function, default_perturb_func
from torch import Tensor

from liga.interpret.common import Interpreter
from liga.type2.common import Type2Explainer, CaptumAttributionWrapper
from liga.torch_extensions.classifier import TorchImageClassifier


class LimeType2Explainer(Type2Explainer):

    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter,
                 interpretable_model: Optional[Model] = None,
                 similarity_func: Optional[Callable] = None,
                 perturb_func: Optional[Callable] = None):
        super().__init__(classifier=classifier,
                         interpreter=interpreter)

        if interpretable_model is None:
            interpretable_model = SkLearnLasso(alpha=1.0)

        if similarity_func is None:
            similarity_func = get_exp_kernel_similarity_function()

        if perturb_func is None:
            perturb_func = default_perturb_func

        self.captum_wrapper = CaptumAttributionWrapper(classifier=classifier,
                                                       attribution_method=LimeBase,
                                                       interpretable_model=interpretable_model,
                                                       similarity_func=similarity_func,
                                                       perturb_func=perturb_func,
                                                       perturb_interpretable_space=True,
                                                       from_interp_rep_transform=self.from_interp_rep_transform,
                                                       to_interp_rep_transform=None)

    @staticmethod
    def from_interp_rep_transform(concept_bits: Tensor, _: Tensor, image: np.ndarray, concept_masks: [np.ndarray],
                                  baseline_rgb: Tuple[float, float, float] = (0., 0., 0.), **kwargs):
        device = concept_bits.device
        concept_bits = (concept_bits
                        .squeeze()
                        .cpu()
                        .detach()
                        .numpy()
                        .astype(np.bool_))
        assert len(concept_bits.shape) == 1
        masked_out_idx = np.bitwise_not(concept_bits)
        concept_masks = np.asarray(concept_masks)
        for mask in concept_masks[masked_out_idx]:
            image[mask] = baseline_rgb
        # image is in shape H x W x C, but torch needs C x H x W
        return torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float)

    @staticmethod
    def _complement_mask(masks: [np.ndarray]):
        return np.bitwise_not(reduce(np.bitwise_or, masks)).astype(np.bool)

    def __call__(self, image: np.ndarray, image_id: Optional[str] = None, **kwargs) -> List[Tuple[int, float]]:
        ids_and_masks = list(self.interpreter(image=image,
                                              image_id=image_id,
                                              **kwargs))
        if len(ids_and_masks) > 0:
            ids, masks = list(zip(*ids_and_masks))
            kwargs.update(image=image,
                          num_interp_features=len(masks) + 1,
                          concept_masks=list(masks) + [self._complement_mask(masks)])
            influences = self.captum_wrapper(image=image,
                                             additional_attribution_args=kwargs)
            # the following zip removes the complement mask automagically
            yield from ((concept_id, influence) for concept_id, influence in zip(ids, influences))

    def __str__(self):
        return 'lime'
