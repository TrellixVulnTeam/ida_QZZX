import abc
from dataclasses import dataclass, field
from typing import Optional, Type, Any, Tuple, Iterable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency, DeepLift, visualization as viz, GradientAttribution
from petastorm.unischema import Unischema
from simple_parsing import ArgumentParser

from simexp.common import LoggingConfig, Classifier
from simexp.describe.common import ImageReadConfig
from simexp.describe.torch_based.base import TorchConfig
from simexp.influence.common import InfluenceEstimator, AnchorInfluenceEstimator, LIMEInfluenceEstimator
from simexp.spark import Field, Schema, PetastormWriteConfig, DictBasedDataGenerator
from simexp.torch_extensions.classifier import TorchImageClassifier, TorchImageClassifierSerialization


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
        attr = algo.attribute(img_tensor, target=int(pred_class))
        attr = np.sum(np.transpose(attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0)), 2)
        attr = attr * (attr > 0)  # we only consider pixels that contribute to the prediction (positive influence value)
        return attr / np.sum(attr)


@dataclass(unsafe_hash=True)
class IntegratedGradientsInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=IntegratedGradients, init=False)


@dataclass(unsafe_hash=True)
class SaliencyInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=Saliency, init=False)


@dataclass(unsafe_hash=True)
class DeepLiftInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=DeepLift, init=False)


@dataclass
class TorchInfluenceGenerator(DictBasedDataGenerator):
    read_cfg: ImageReadConfig
    output_schema: Unischema = field(default=Schema.PIXEL_INFLUENCES, init=False)

    classifier_serial: TorchImageClassifierSerialization
    torch_cfg: TorchConfig

    influence_estimators: List[InfluenceEstimator]

    image_size: Tuple[int, int] = (224, 224)

    # if not None, sample the given number of observations from each class instead
    # of reading the dataset once front to end.
    observations_per_class: Optional[int] = None

    # whether to visualize the influential pixels for each image and estimator
    debug: bool = False

    # after how many observations to output a log message with hit frequencies of the different estimators
    hit_freq_logging: int = 30

    def __post_init__(self):
        # load the classifier from its name
        @dataclass
        class PartialTorchImageClassifier(TorchImageClassifier):
            torch_cfg: TorchConfig = field(default_factory=lambda: self.torch_cfg, init=False)

        self.classifier = PartialTorchImageClassifier.load(self.classifier_serial.path)

    def generate(self) -> Iterable[Dict[str, Any]]:
        total_count = 0
        count_per_class = np.zeros((self.classifier.num_classes,))

        with self.read_cfg.make_reader(None) as reader:
            for row in reader:
                pred = np.uint16(np.argmax(self.classifier.predict_proba(np.expand_dims(row.image, 0))[0]))

                if self.observations_per_class is not None:
                    if count_per_class[pred] >= self.observations_per_class:
                        self._log_item('Skipping, we already have enough observations of class {}.'.format(pred))
                        continue
                    elif np.sum(count_per_class) >= self.classifier.num_classes * self.observations_per_class:
                        self._log_item('Stopping, we now have enough observations of all classes.')
                        break

                count_per_class[pred] += 1
                total_count += 1

                for influence_estimator in self.influence_estimators:
                    influence_mask = influence_estimator.get_influence_mask(self.classifier, row.image, pred)
                    if self.debug:
                        fig = plt.figure()
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)
                        viz.visualize_image_attr(np.expand_dims(influence_mask, 2), row.image,
                                                 sign='positive', method='blended_heat_map', use_pyplot=False,
                                                 plt_fig_axis=(fig, ax))

                    yield {Field.IMAGE_ID.name: row.image_id,
                           Field.INFLUENCE_MASK.name: influence_mask,
                           Field.PREDICTED_CLASS.name: pred,
                           Field.INFLUENCE_ESTIMATOR.name: str(influence_estimator)}


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
        self.describer.spark_cfg.write_petastorm(self.describer.to_df(), self.write_cfg)


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
