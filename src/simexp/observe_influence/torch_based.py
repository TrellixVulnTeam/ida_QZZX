import abc
from dataclasses import dataclass, field
from typing import Optional, Type, List, Dict, Any

import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency, DeepLift, GradientAttribution
from petastorm.unischema import Unischema
from simple_parsing import ArgumentParser

from liga.common import LoggingConfig, Classifier
from simexp.observe_influence.common import InfluenceEstimator, AnchorInfluenceEstimator, LIMEInfluenceEstimator, \
    InfluenceGenerator
from simexp.spark import Schema, PetastormWriteConfig
from liga.torch_extensions.classifier import TorchImageClassifier, TorchImageClassifierLoader


@dataclass(unsafe_hash=True)
class CaptumInfluenceEstimator(InfluenceEstimator, abc.ABC):
    """
    Wrapper for influence estimators from the `captum` package.
    """

    algorithm: Type[GradientAttribution]

    def get_influence_mask(self, classifier: Classifier, img: np.ndarray, pred_class: np.uint16) -> np.ndarray:
        if not isinstance(classifier, TorchImageClassifier):
            raise NotImplementedError('The captum algorithms only work for torch classifiers.')

        img_tensor = torch.from_numpy(img).float().to(classifier.torch_cfg.device).permute(2, 0, 1).unsqueeze(0)
        algo = self.algorithm(classifier.torch_model)
        attr = algo.attribute(img_tensor, target=int(pred_class), **self._attribution_extra_args())
        return np.sum(np.transpose(attr.squeeze(0).cpu().detach().numpy().astype(np.float_), (1, 2, 0)), 2)

    @staticmethod
    def _attribution_extra_args() -> Dict[str, Any]:
        return {}


@dataclass(unsafe_hash=True)
class IntegratedGradientsInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=IntegratedGradients, init=False)


@dataclass(unsafe_hash=True)
class SaliencyInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=Saliency, init=False)

    @staticmethod
    def _attribution_extra_args() -> Dict[str, Any]:
        return {'abs': False}


@dataclass(unsafe_hash=True)
class DeepLiftInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=DeepLift, init=False)


@dataclass
class TorchInfluenceGenerator(TorchImageClassifierLoader, InfluenceGenerator):

    name: str = field(default=None, init=False)

    observations_per_class: Optional[int] = None
    debug: bool = False
    hit_freq_logging: int = 30

    def __post_init__(self):
        super().__post_init__()
        self.name = 'pixel_influences({})'.format(self.classifier.name)


@dataclass
class CLTorchInfluenceGenerator(TorchInfluenceGenerator):
    influence_estimators: List[InfluenceEstimator] = field(default=list, init=False)


@dataclass
class InfluenceWriteConfig(PetastormWriteConfig):
    output_schema: Unischema = field(default=Schema.PIXEL_INFLUENCES, init=False)


@dataclass
class CLInterface:
    write_cfg: InfluenceWriteConfig
    describer: CLTorchInfluenceGenerator

    lime_ie: LIMEInfluenceEstimator
    anchor_ie: AnchorInfluenceEstimator
    igrad_ie: IntegratedGradientsInfluenceEstimator
    saliency_ie: SaliencyInfluenceEstimator
    deep_lift_ie: DeepLiftInfluenceEstimator

    # which influence estimators to use
    used_influence_estimators: List[str] = field(default_factory=list)

    # after which number of observations to 'flush', i.e., append to the petastorm store.
    # use this if you generate a very large dataset that does not fit into RAM.
    flush_batch_size: Optional[int] = None

    def __post_init__(self):
        ies = {'deeplift': self.deep_lift_ie,
               'saliency': self.saliency_ie,
               'igrad': self.igrad_ie,
               'anchor': self.anchor_ie,
               'lime': self.lime_ie}
        if not self.used_influence_estimators:
            self.describer.influence_estimators = ies.values()  # use all
        else:
            self.describer.influence_estimators = [ies[k] for k in self.used_influence_estimators]

    def run(self):
        self.describer.write_petastorm(self.write_cfg, self.flush_batch_size)


if __name__ == '__main__':
    """
    Compute the influence of each pixel of an image on the output of an image classifier.
    The images are read from a petastorm parquet store. In this parquet store,
    there must be three fields:

      - A field holding a unique id for each image.
        This field is specified by the caller of this method as `id_field`.
      - A field holding image data encoded with the petastorm png encoder.
        This field must be named *image*.

    Use one of the implementations in the submodules of this package.
    """
    parser = ArgumentParser(description='Compute pixel influences.')
    parser.add_arguments(CLInterface, dest='task')
    parser.add_arguments(LoggingConfig, dest='logging')
    args = parser.parse_args()
    args.task.run()
