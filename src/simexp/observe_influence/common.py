import abc
from dataclasses import dataclass, field
from typing import Optional, List, Iterator

import matplotlib.pyplot as plt
import numpy as np
from anchor import anchor_image
from captum.attr import visualization as viz
from lime import lime_image
from petastorm.unischema import Unischema

from liga.common import Classifier, RowDict
from simexp.describe.common import ImageReadConfig
from simexp.spark import DictBasedDataGenerator, Schema, Field


@dataclass(unsafe_hash=True)
class InfluenceEstimator(abc.ABC):
    """
    A method to estimate the most influential pixels for a given classifier prediction.
    """

    @abc.abstractmethod
    def get_influence_mask(self, classifier: Classifier, img: np.ndarray, pred_class: np.uint16) -> np.ndarray:
        """
        Returns a float numpy array in the dimensions of the input `img` whose entries
        represent the influence of each pixel on the classification `pred_class` by `classifier`.
        """
        pass


@dataclass
class InfluenceGenerator(DictBasedDataGenerator):

    # where to read images from
    read_cfg: ImageReadConfig

    # output schema of this generator
    output_schema: Unischema = field(default=Schema.PIXEL_INFLUENCES, init=False)

    # which classifier to observe
    classifier: Classifier

    # which influence estimators to use
    influence_estimators: List[InfluenceEstimator]

    # if not None, sample the given number of observations from each class instead
    # of reading the dataset once front to end.
    observations_per_class: Optional[int]

    # whether to visualize the influential pixels for each image and estimator
    debug: bool

    # after how many observations to output a log message with hit frequencies of the different estimators
    hit_freq_logging: int

    def generate(self) -> Iterator[RowDict]:
        total_count = 0
        count_per_class = np.zeros((self.classifier.num_classes,))

        with self.read_cfg.make_reader(None) as reader:
            for row in reader:
                pred = self.classifier.predict_single(row.image)

                if self.observations_per_class is not None:
                    if count_per_class[pred] >= self.observations_per_class:
                        self.log_item('Skipping, we already have enough observations of class {}.'.format(pred))
                        continue
                    elif np.sum(count_per_class) >= self.classifier.num_classes * self.observations_per_class:
                        self.log_item('Stopping, we now have enough observations of all classes.')
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
                                                 sign='all', method='blended_heat_map', use_pyplot=False,
                                                 plt_fig_axis=(fig, ax))
                        plt.show()
                        input('--- showing influences from {}.\n'
                              'press enter to continue ---'.format(influence_estimator))
                        plt.clf()

                    yield {Field.IMAGE_ID.name: row.image_id,
                           Field.INFLUENCE_MASK.name: influence_mask,
                           Field.PREDICTED_CLASS.name: pred,
                           Field.INFLUENCE_ESTIMATOR.name: str(influence_estimator)}


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

        return influence_mask.astype(np.float_) / np.count_nonzero(influence_mask)


@dataclass(unsafe_hash=True)
class LIMEInfluenceEstimator(InfluenceEstimator):
    """
    Uses the LIME approach
    http://doi.acm.org/10.1145/2939672.2939778
    for determining influential pixels.
    """
    search_num_features: int = 100000
    num_samples: int = 1000

    positive_only: bool = False
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

        mask = np.zeros(explanation.segments.shape, dtype=np.float_)
        for feature, weight in explanation.local_exp:
            mask[explanation.segments == feature] = weight
        return mask
