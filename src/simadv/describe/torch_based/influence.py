import abc
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients, Saliency, DeepLift, visualization as viz, GradientAttribution
from lime import lime_image

from typing import Optional, Type, Any, Tuple, Iterable, Dict, List

import numpy as np
from petastorm.unischema import Unischema
from simple_parsing import ArgumentParser

from simadv.common import LoggingConfig, Classifier
from simadv.io import Field, Schema, PetastormWriteConfig, SparkSessionConfig
from simadv.describe.torch_based.base import TorchConfig, TorchImageClassifier, TorchImageClassifierSerialization

from anchor import anchor_image

from simadv.describe.common import DictBasedImageDescriber, ImageReadConfig


@dataclass(unsafe_hash=True)
class InfluenceEstimator(abc.ABC):
    """
    A method to estimate the most influential pixels for a given classifier prediction.
    """

    @abc.abstractmethod
    def get_influence_mask(self, classifier: Classifier, img: np.ndarray, pred_class: np.uint16) -> np.ndarray:
        """
        Returns a float numpy array in the dimensions of the input `img` whose entries sum up to 1
        and represent the influence of each pixel on the classification `pred_class` by `classifier`.
        """
        pass


@dataclass(unsafe_hash=True)
class AnchorInfluenceEstimator(InfluenceEstimator):
    """
    Uses the anchor approach
    https://ojs.aaai.org/index.php/AAAI/article/view/11491
    for determining influential pixels.
    """

    beam_size: int = 1
    coverage_samples: int = 10000
    stop_on_first: bool = False
    threshold: float = 0.95
    delta: float = 0.1
    tau: float = 0.15
    batch_size: int = 100
    max_anchor_size: Optional[int] = None

    def __post_init__(self):
        self.anchor = anchor_image.AnchorImage()

    def get_influence_mask(self, classifier: Classifier, img: np.ndarray, pred_class: np.uint16) -> np.ndarray:
        explanation = self.anchor.explain_instance(img,
                                                   classifier.predict_proba,
                                                   beam_size=self.beam_size,
                                                   threshold=self.threshold,
                                                   max_anchor_size=self.max_anchor_size,
                                                   coverage_samples=self.coverage_samples,
                                                   stop_on_first=self.stop_on_first,
                                                   tau=self.tau,
                                                   delta=self.delta,
                                                   batch_size=self.batch_size)
        # explanation has format (segmentation_mask, [(segment_id, '', mean, [negatives], 0)])
        # where
        #    segmentation mask: assigns each pixel a segment id
        #    segment_id: identifies a segment
        #    mean: ?
        #    negatives: the strongest (?) counter examples, i.e., images where the anchor fails
        segmentation_mask, relevant_segments = explanation

        influence_mask = np.zeros(segmentation_mask.shape, dtype=np.int)
        for segment_id, _, _, _, _ in relevant_segments:
            influence_mask = np.bitwise_or(influence_mask,
                                           segmentation_mask == segment_id)

        if not np.any(influence_mask):
            return np.ones(influence_mask.shape) / np.size(influence_mask)

        return influence_mask.astype(np.float) / np.float(np.count_nonzero(influence_mask))


@dataclass(unsafe_hash=True)
class LIMEInfluenceEstimator(InfluenceEstimator):
    """
    Uses the LIME approach
    http://doi.acm.org/10.1145/2939672.2939778
    for determining influential pixels.
    """
    search_num_features: int = 100000
    num_samples: int = 1000

    positive_only: bool = True
    negative_only: bool = False
    hide_rest: bool = False
    explain_num_features: int = 5
    min_weight: float = 0.

    def __post_init__(self):
        self.lime = lime_image.LimeImageExplainer()

    def get_influence_mask(self, classifier: Classifier, img: np.ndarray, pred_class: np.uint16) -> np.ndarray:
        explanation = self.lime.explain_instance(img, classifier.predict_proba,
                                                 labels=(pred_class,),
                                                 top_labels=None,
                                                 num_features=self.search_num_features,
                                                 num_samples=self.num_samples)
        _, influence_mask = explanation.get_image_and_mask(pred_class,
                                                           positive_only=self.positive_only,
                                                           negative_only=self.negative_only,
                                                           num_features=self.explain_num_features,
                                                           min_weight=self.min_weight)

        if not np.any(influence_mask):
            return np.ones(influence_mask.shape) / np.size(influence_mask)

        return influence_mask.astype(np.float) / np.float(np.count_nonzero(influence_mask))


@dataclass(unsafe_hash=True)
class CaptumInfluenceEstimator(InfluenceEstimator, abc.ABC):
    """
    Wrapper for influence estimators from the `captum` package.
    """

    algorithm: Type[GradientAttribution]

    def get_influence_mask(self, classifier: Classifier, img: np.ndarray, pred_class: np.uint16) -> np.ndarray:
        if not isinstance(classifier, TorchImageClassifier):
            raise NotImplementedError('The captum algorithms only work for torch_based classifiers.')

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
class TorchInfluenceImageDescriber(DictBasedImageDescriber):
    read_cfg: ImageReadConfig
    output_schema: Unischema = field(default=Schema.PIXEL_INFLUENCES, init=False)

    classifier_serial: TorchImageClassifierSerialization
    torch_cfg: TorchConfig

    influence_estimators: List[InfluenceEstimator]

    image_size: Tuple[int, int] = (224, 224)

    concept_group_name: str = field(default='pixel_influences', init=False)

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
class CLTorchInfluenceImageDescriber(TorchInfluenceImageDescriber):
    influence_estimators: List[InfluenceEstimator] = field(default=list, init=False)


@dataclass
class InfluenceWriteConfig(PetastormWriteConfig):
    output_schema: Unischema = field(default=Schema.PIXEL_INFLUENCES, init=False)


@dataclass
class CLInterface:
    write_cfg: InfluenceWriteConfig
    describer: CLTorchInfluenceImageDescriber

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
