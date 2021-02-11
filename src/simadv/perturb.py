import abc
import logging
from dataclasses import dataclass, field
from typing import Iterable, Tuple, Any, Iterator

import numpy as np
import itertools as it

from petastorm.reader import Reader
from petastorm.unischema import Unischema

from simadv.common import RowDict
from simadv.spark import Field, Schema, PetastormReadConfig, PetastormWriteConfig


class Perturber(abc.ABC):

    @abc.abstractmethod
    def perturb(self, influential_counts: np.ndarray, counts: np.ndarray, sampler: Iterable[Tuple[np.ndarray, Any]]) \
            -> Iterable[Tuple[np.ndarray, Any]]:
        """
        Takes in two arrays `counts` and `influential_counts` of the same dimension 1xO,
        where O is the number of objects in a classification task.
        `counts` are object counts on an image, and `influential_counts` represents a subset of these objects.
        This subset comprises objects that one deems influential for the classification of this image.

        From this subset the method derives alternative count arrays that by expectation
        all yield the same prediction as `count`.
        The method can use the `sampler` to draw random object count arrays together with their image id
        from the same image distribution that `count` was derived from.

        :return tuples of count arrays and image ids. if a count array was derived from an image drawn from `sampler`,
            the corresponding image id must be returned, else None.
        """


@dataclass
class LocalPerturber(Perturber):
    """
    Assumes that given "influential objects" on an image are a locally sufficient condition for its classification,
    i.e., for images that are similar.
    Derives object counts for "similar" images by dropping all other, "non-influential" objects from the given image.
    Generates all combinations of dropped object counts if they are less than `max_perturbations`, else generates
    `max_perturbations` combinations randomly.
    """

    # upper bound for perturbations to generate
    max_perturbations: int = 10

    def perturb(self, influential_counts: np.ndarray, counts: np.ndarray, sampler: Iterable[Tuple[np.ndarray, Any]]) \
            -> Iterable[Tuple[np.ndarray, Any]]:
        droppable_counts = counts - influential_counts

        if np.multiply.reduce(droppable_counts + 1) > self.max_perturbations:
            yield from self._perturb_random(counts, droppable_counts)
        else:
            yield from self._perturb_exhaustive(counts, droppable_counts)

    @staticmethod
    def _perturb_exhaustive(counts, droppable_counts):
        gens = []
        for droppable_index in np.flatnonzero(droppable_counts):
            gens.append(zip(range(0, droppable_counts[droppable_index] + 1), it.repeat(droppable_index)))

        for drops in it.product(*gens):
            perturbed = counts.copy()
            for drop_count, drop_index in drops:
                perturbed[drop_index] -= drop_count
            yield perturbed, None

    def _perturb_random(self, counts, droppable_counts):
        for _ in range(self.max_perturbations):
            perturbed = counts.copy()
            for drop_index in np.flatnonzero(droppable_counts):
                drop_count = np.random.randint(droppable_counts[drop_index] + 1)
                perturbed[drop_index] -= drop_count
            yield perturbed, None


@dataclass
class GlobalPerturber(Perturber):
    """
    Assumes that given "influential objects" on an image are a globally sufficient condition for its classification.
    Hence replaces all other objects randomly with objects from other images from the same distribution,
    and assumes that the classification stays the same.
    """

    # how many perturbed object counts to generate for each image
    num_perturbations: int = 10

    def perturb(self, influential_counts: np.ndarray, counts: np.ndarray, sampler: Iterable[Tuple[np.ndarray, Any]]) \
            -> Iterable[Tuple[np.ndarray, Any]]:
        """
        Returns the original counts array plus `num_perturbations` (class parameter) additional arrays.
        The latter are unions of `influential_counts` with random counts drawn from `sampler`.
        """
        # original counts
        yield counts, None

        if np.any(influential_counts):
            # sample random object counts from the same distribution
            for sample_counts, sample_id in it.islice(sampler, self.num_perturbations):
                # keep all influential objects from the original image,
                # change the rest based on the sample → pairwise maximum of counts
                combined_counts = np.maximum(influential_counts, sample_counts)
                yield combined_counts, sample_id


class BoxInfluenceDetector(abc.ABC):

    @abc.abstractmethod
    def detect(self, influence_mask: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> bool:
        """
        Decides whether the bounding box defined by `x_min`, `x_max`, `y_min` and `y_max`
        is influential for a classification according to `influence_mask`.
        """


@dataclass
class LiftBoxInfluenceDetector(BoxInfluenceDetector):

    # an object is considered relevant for the prediction of the classifier
    # if the sum of influence values in the objects bounding box
    # exceeds the fraction 'object area' / 'image area' by a factor lift_threshold`.
    lift_threshold: float = 1.5

    def detect(self, influence_mask: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> bool:
        height, width = influence_mask.shape
        img_area = float(height * width)

        x_min = int(np.floor(x_min * width))  # in case of "rounding doubt" let the object be larger
        x_max = int(np.ceil(x_max * width))
        y_min = int(np.floor(y_min * height))
        y_max = int(np.ceil(y_max * height))
        box_area = float((y_max - y_min) * (x_max - x_min))
        # lift = how much more influence than expected do pixels of the box have?
        lift = np.sum(influence_mask[y_min:y_max, x_min:x_max]) / (box_area / img_area)
        return lift > self.lift_threshold


@dataclass
class PerturbReadConfig(PetastormReadConfig):
    input_schema: Unischema = field(default=Schema.PIXEL_INFLUENCES, init=False)


@dataclass
class PerturbWriteConfig(PetastormWriteConfig):
    output_schema: Unischema = field(default=Schema.PERTURBED_CONCEPT_COUNTS, init=False)


@dataclass
class PerturbTask:
    read_cfg: PerturbReadConfig
    write_cfg: PerturbWriteConfig

    # which detectors to use
    detectors: [BoxInfluenceDetector]

    # which perturbers to use
    perturbers: [Perturber]

    # from where to sample additional images with object bounding boxes
    sampling_read_cfg: PetastormReadConfig

    # the number of object classes annotated for the input images
    num_objects: int

    def sampler(self, sampling_reader: Reader):
        for sampled_row in sampling_reader:
            sample_counts = np.zeros((self.num_objects,), dtype=np.uint8)
            for box in sampled_row.boxes:
                obj_id = box[0]
                sample_counts[obj_id] += 1
            yield sample_counts, sampled_row.image_id

    def generate(self) -> Iterator[RowDict]:
        with self.read_cfg.make_reader(None) as reader:
            with self.sampling_read_cfg.make_reader([Field.IMAGE_ID.name, Field.CONCEPT_NAMES.name,
                                                     Field.CONCEPT_MASKS.name], num_epochs=None) as sampling_reader:
                logging.info('Start reading images...')
                for row in reader:

                    counts = np.zeros((self.num_objects,), dtype=np.uint8)
                    for concept_name in row.concept_names:
                        counts[concept_name] += 1

                    yield {Field.CONCEPT_COUNTS.name: counts,
                           Field.PREDICTED_CLASS.name: row.predicted_class,
                           Field.INFLUENCE_ESTIMATOR.name: None,
                           Field.PERTURBER.name: None,
                           Field.DETECTOR.name: None,
                           Field.IMAGE_ID.name: row.image_id,
                           Field.PERTURBED_IMAGE_ID: None}

                    for detector in self.detectors:
                        min_counts = np.zeros((self.num_objects,), dtype=np.uint8)
                        for box in row.boxes:
                            obj_id, x_min, x_max, y_min, y_max = box[:5]

                            if detector.detect(row.mask, x_min, x_max, y_min, y_max):
                                min_counts[obj_id] += 1
                                logging.info('[{}] Object {} has exceptional influence.'
                                             .format(row.influence_estimator, obj_id))

                        for perturber in self.perturbers:
                            for perturbed_counts, perturbed_image_id \
                                    in perturber.perturb(min_counts, counts, self.sampler(sampling_reader)):
                                yield {Field.CONCEPT_COUNTS.name: perturbed_counts,
                                       Field.PREDICTED_CLASS.name: row.predicted_class,
                                       Field.INFLUENCE_ESTIMATOR.name: row.influence_estimator,
                                       Field.PERTURBER.name: perturber,
                                       Field.DETECTOR.name: detector,
                                       Field.IMAGE_ID.name: row.image_id,
                                       Field.PERTURBED_IMAGE_ID: perturbed_image_id}

    def run(self):
        self.write_cfg.write_parquet(self.generate())
