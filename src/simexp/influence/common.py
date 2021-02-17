import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
from anchor import anchor_image
from lime import lime_image

from simexp.common import Classifier


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