import abc
import logging
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients, Saliency, DeepLift, visualization as viz, GradientAttribution
from lime import lime_image

from typing import Optional, Type, Any, Tuple, Iterable, Dict, List

import numpy as np
from petastorm.unischema import Unischema
from simple_parsing import ArgumentParser

from simadv.common import LoggingConfig, TorchConfig, \
    TorchImageClassifier, TorchImageClassifierSerialization, Classifier, \
    PetastormReadConfig, Schema, PetastormWriteConfig, Field

from anchor import anchor_image


class InfluenceEstimator(abc.ABC):
    """
    A method to estimate the most influential pixels for a given classifier prediction.
    """

    @property
    @abc.abstractmethod
    def abbreviation(self):
        pass

    @abc.abstractmethod
    def get_influence_mask(self, classifier: Classifier, img: np.ndarray, pred_class: np.uint16) -> np.ndarray:
        """
        Returns a float numpy array in the dimensions of the input `img` whose entries sum up to 1
        and represent the influence of each pixel on the classification `pred_class` by `classifier`.
        """
        pass


@dataclass
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

    @property
    def abbreviation(self):
        return 'anchor'

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


@dataclass
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

    @property
    def abbreviation(self):
        return 'lime'

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


@dataclass
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


@dataclass
class IntegratedGradientsInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=IntegratedGradients, init=False)

    @property
    def abbreviation(self):
        return 'igrad'


@dataclass
class SaliencyInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=Saliency, init=False)

    @property
    def abbreviation(self):
        return 'saliency'


@dataclass
class DeepLiftInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=DeepLift, init=False)

    @property
    def abbreviation(self):
        return 'deeplift'


@dataclass
class InfluenceReadConfig(PetastormReadConfig):
    input_schema: Unischema = field(default=Schema.OBJECT_BOXES, init=False)


@dataclass
class InfluenceWriteConfig(PetastormWriteConfig):
    output_schema: Unischema = field(default=Schema.PIXEL_INFLUENCES, init=False)


@dataclass
class TorchInfluenceTask:
    read_cfg: InfluenceReadConfig
    write_cfg: InfluenceWriteConfig

    classifier_serial: TorchImageClassifierSerialization
    torch_cfg: TorchConfig

    lime_ie: LIMEInfluenceEstimator
    anchor_ie: AnchorInfluenceEstimator
    igrad_ie: IntegratedGradientsInfluenceEstimator
    saliency_ie: SaliencyInfluenceEstimator
    deep_lift_ie: DeepLiftInfluenceEstimator

    image_size: Tuple[int, int] = (224, 224)

    # which influence estimators to use
    used_influence_estimators: List[str] = field(default_factory=list)

    # an object is considered relevant for the prediction of the classifier
    # if the sum of influence values in the objects bounding box
    # exceeds the fraction 'object area' / 'image area' by a factor lift_threshold`.
    lift_threshold: float = 1.5

    # if not None, sample the given number of observations from each class instead
    # of reading the dataset once front to end.
    observations_per_class: Optional[int] = None

    # after which time to automatically stop
    time_limit_s: Optional[int] = None

    # whether to visualize the influential pixels for each image and explainer
    debug: bool = False

    def __post_init__(self):
        # load the classifier from its name
        @dataclass
        class PartialTorchImageClassifier(TorchImageClassifier):
            torch_cfg: TorchConfig = field(default_factory=lambda: self.torch_cfg, init=False)

        self.classifier = PartialTorchImageClassifier.load(self.classifier_serial.path)
        self.influence_estimators = (self.deep_lift_ie,
                                     self.saliency_ie,
                                     self.igrad_ie,
                                     self.anchor_ie,
                                     self.lime_ie)

        unknown_ies = set(self.used_influence_estimators) - set(est.abbreviation for est in self.influence_estimators)
        if unknown_ies:
            raise ValueError('Unknown influence estimators: {}'.format(unknown_ies))

    def generate(self) -> Iterable[Dict[str, Any]]:
        last_time = start_time = time.time()
        total_count = 0
        hit_counts = {est: 0 for est in self.influence_estimators}

        count_per_class = np.zeros((self.classifier.task.num_classes,))

        with self.read_cfg.make_reader(None) as reader:
            logging.info('Start reading images...')
            for row in reader:
                pred = np.uint16(np.argmax(self.classifier.predict_proba(np.expand_dims(row.image, 0))[0]))

                if self.observations_per_class is not None:
                    if count_per_class[pred] >= self.observations_per_class:
                        logging.info('Skipping, we already have enough observations of class {}.'.format(pred))
                        continue
                    elif np.sum(count_per_class) >= self.classifier.task.num_classes * self.observations_per_class:
                        logging.info('Stopping, we now have enough observations of all classes.')
                        break

                count_per_class[pred] += 1
                total_count += 1

                for influence_estimator in self.influence_estimators:
                    if self.used_influence_estimators \
                            and influence_estimator not in self.used_influence_estimators:
                        continue

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
                           Field.BOXES.name: row.boxes[:, :5],
                           Field.INFLUENCE_MASK.name: influence_mask,
                           Field.PREDICTED_CLASS.name: pred,
                           Field.INFLUENCE_ESTIMATOR.name: influence_estimator}

                current_time = time.time()
                if current_time - last_time > 5:
                    f = ', '.join(['{}: {:.2f}'.format(est, float(hit_counts[est]) / total_count)
                                  for est in self.influence_estimators])
                    logging.info('Processed {} images so far. Hit frequencies: {}'
                                 .format(total_count, f))
                    last_time = current_time

                if self.time_limit_s is not None and current_time - start_time > self.time_limit_s:
                    logging.info('Reached timeout! Stopping.')
                    break

    def run(self):
        self.write_cfg.write_parquet(self.generate())


if __name__ == '__main__':
    """
    Compute the influence of each pixel of an image on the output of an image classifier.
    The images are read from a petastorm parquet store. In this parquet store,
    there must be three fields:

      - A field holding a unique id for each image.
        This field is specified by the caller of this method as `id_field`.
      - A field holding image data encoded with the petastorm png encoder.
        This field must be named *image*.
      - A field holding bounding boxes of visible objects on the image.
        This field must be named *boxes*.

    Use one of the implementations in the submodules of this package.
    """
    parser = ArgumentParser(description='Compute pixel influences.')
    parser.add_arguments(TorchInfluenceTask, dest='task')
    parser.add_arguments(LoggingConfig, dest='logging')
    args = parser.parse_args()
    args.task.run()
