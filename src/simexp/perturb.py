import abc
import itertools as it
from dataclasses import dataclass, field
from typing import Iterable, Tuple, Any, List

import numpy as np
import pyspark.sql.functions as sf
import pyspark.sql.types as st
from petastorm.codecs import ScalarCodec
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import DataFrame
from simple_parsing import ArgumentParser

from simexp.common import LoggingConfig
from simexp.spark import Field, SparkSessionConfig, PetastormWriteConfig, ConceptMasksUnion, \
    DataGenerator


class AutoRepr:
    def __repr__(self):
        items = ('{}={}'.format(k, v) for k, v in self.__dict__.items())
        return '{}({})>'.format(self.__class__.__name__, ', '.join(items))


class Perturber(AutoRepr, abc.ABC):

    @abc.abstractmethod
    def perturb(self, influential_counts: np.ndarray, counts: np.ndarray, sampler: Iterable[Tuple[str, np.ndarray]]) \
            -> Iterable[Tuple[np.ndarray, Any]]:
        """
        Takes in two arrays `counts` and `influential_counts` of the same dimension 1xO,
        where O is the number of objects in a classification task.
        `counts` are concept counts on an image, and `influential_counts` represents a subset of these concepts.
        This subset comprises concepts that one deems influential for the classification of this image.

        From this subset the method derives alternative count arrays that by expectation
        all yield the same prediction as `count`.
        The method can use the `sample_queue` to draw random concept count arrays together with their image id
        from the same image distribution that `count` was derived from.

        :return tuples of count arrays and image ids. if a count array was derived from an image drawn from `sampler`,
            the corresponding image id must be returned, else None.
        """


class LocalPerturber(Perturber):
    """
    Assumes that given "influential concepts" on an image are a locally sufficient condition for its classification,
    i.e., for images that are similar.
    Derives concept counts for "similar" images by dropping all other, "non-influential" concepts from the given image.
    Generates all combinations of dropped concept counts if they are less than `max_perturbations`, else generates
    `max_perturbations` combinations randomly.
    """

    def __init__(self, max_perturbations: int = 10):
        """
        :param max_perturbations: upper bound for perturbations to generate
        """
        self.max_perturbations = max_perturbations

    def perturb(self, influential_counts: np.ndarray, counts: np.ndarray, sampler: Iterable[Tuple[str, np.ndarray]]) \
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


class GlobalPerturber(Perturber):
    """
    Assumes that given "influential concepts" on an image are a globally sufficient condition for its classification.
    Hence replaces all other concepts randomly with concepts from other images from the same distribution,
    and assumes that the classification stays the same.
    """

    def __init__(self, num_perturbations: int = 10):
        """
        :param num_perturbations: how many perturbed concept counts to generate for each image
        """
        self.num_perturbations = num_perturbations

    def perturb(self, influential_counts: np.ndarray, counts: np.ndarray, sampler: Iterable[Tuple[str, np.ndarray]]) \
            -> Iterable[Tuple[np.ndarray, Any]]:
        """
        Returns the original counts array plus `num_perturbations` (class parameter) additional arrays.
        The latter are unions of `influential_counts` with random counts drawn from `sampler`.
        """
        # original counts
        yield counts, None

        if np.any(influential_counts):
            # sample random object counts from the same distribution
            for sample_id, sample_counts in it.islice(sampler, 0, self.num_perturbations):
                # keep all influential objects from the original image,
                # change the rest based on the sample → pairwise maximum of counts
                combined_counts = np.maximum(influential_counts, sample_counts)
                yield combined_counts, sample_id


class InfluenceDetector(AutoRepr, abc.ABC):

    @abc.abstractmethod
    def detect(self, influence_mask: np.ndarray, concept_mask: np.ndarray) -> bool:
        """
        Decides whether the mask defined by the boolean matrix `concept_mask`
        is influential for a classification according to the influential pixels
        defined by the float matrix `influence_mask`.
        """


class LiftInfluenceDetector(InfluenceDetector):

    def __init__(self, lift_threshold: float = 1.5):
        """
        :param lift_threshold: a concept is considered relevant for the prediction of the classifier
            if the sum of influence values falling into the mask of the concept
            exceeds the fraction 'concept area' / 'image area' by a factor `lift_threshold`.
        """
        self.lift_threshold = lift_threshold

    def detect(self, influence_mask: np.ndarray, concept_mask: np.ndarray) -> bool:
        height, width = influence_mask.shape
        img_area = float(height * width)

        mask_area = float(np.count_nonzero(concept_mask))
        # lift = how much more influence than expected do pixels of the mask have?
        lift = np.sum(influence_mask[concept_mask]) / (mask_area / img_area)
        return lift > self.lift_threshold


class ConceptCountsSamplingIterator:
    """
    Iterates over random image ids and corresponding concept counts.
    """

    def __init__(self, id_samples: np.ndarray, counts_samples: np.ndarray):
        assert len(id_samples) == len(counts_samples)
        self.id_samples = id_samples
        self.counts_samples = counts_samples

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.randint(0, len(self.counts_samples))
        yield self.id_samples[idx], self.counts_samples[idx]


@dataclass
class PerturbedConceptCountsGenerator(ConceptMasksUnion, DataGenerator):

    # the output schema of this data generator.
    output_schema: Unischema = field(init=False)

    # url of petastorm parquet store of schema `Schema.PIXEL_INFLUENCES`,
    # as generated by the module `observe_influence`
    influences_url: str

    # which detectors to use
    detectors: List[InfluenceDetector]

    # which perturbers to use
    perturbers: List[Perturber]

    # size of the queue of image samples that perturbers use.
    # this should be at least the number of spark executors.
    sampling_queue_size: int = 80

    def __post_init__(self):
        super().__post_init__()

        influence_fields = [Field.IMAGE_ID, Field.PREDICTED_CLASS, Field.INFLUENCE_ESTIMATOR, Field.PERTURBER,
                            Field.DETECTOR, Field.PERTURBED_IMAGE_ID]
        concept_fields = [UnischemaField(concept_name, np.uint8, (), ScalarCodec(st.IntegerType()), False)
                          for concept_name in self.all_concept_names]
        self.output_schema = Unischema('PerturbedConceptCounts', influence_fields + concept_fields)

        id_samples, counts_samples = self._get_sampling_data()
        self.id_samples_broadcast = self.spark_cfg.session.sparkContext.broadcast(id_samples)
        self.counts_samples_broadcast = self.spark_cfg.session.sparkContext.broadcast(counts_samples)

    def __getstate__(self):
        # note: we only return the state necessary for the method `_perturb_concepts_on_image`.
        # other attributes will be lost.
        return self.all_concept_names, self.detectors, self.perturbers, self.output_schema

    def __setstate__(self, state):
        self.all_concept_names, self.detectors, self.perturbers, self.output_schema = state

    def _get_sampling_data(self) -> Tuple[np.ndarray, np.ndarray]:
        id_samples = []
        counts_samples = []

        with self._log_task('Shuffling concept masks for random sampling'):
            for image_row in self.union_df.collect():
                image_id = Field.IMAGE_ID.decode(image_row[Field.IMAGE_ID.name])
                counts = self._get_counts(image_row)
                id_samples.append(image_id)
                counts_samples.append(counts)

        return np.asarray(id_samples), np.asarray(counts_samples)

    def _get_influences_df(self):
        return self.spark_cfg.session.read.parquet(self.influences_url) \
            .groupBy(Field.IMAGE_ID.name) \
            .agg(*[sf.collect_list(sf.col(f.name)).alias(f.name)
                   for f in [Field.PREDICTED_CLASS, Field.INFLUENCE_ESTIMATOR, Field.INFLUENCE_MASK]])

    def _get_counts(self, per_image_row: st.Row) -> np.ndarray:
        counts = np.zeros((len(self.all_concept_names, )), dtype=np.uint8)

        # each image has multiple arrays of concept names from different describers
        for concept_names in per_image_row[Field.CONCEPT_NAMES.name]:
            for concept_name in concept_names:
                counts[self.all_concept_names.index(concept_name)] += 1

        return counts

    @staticmethod
    def _get_predicted_class(per_image_row: st.Row) -> int:
        # classification should be constant for the same image
        predicted_classes = set(per_image_row[Field.PREDICTED_CLASS.name])
        assert len(predicted_classes) == 1
        return predicted_classes.pop()

    def _perturb_concepts_on_image(self, per_image_row: st.Row):
        image_id = Field.IMAGE_ID.decode_from_row_dict(per_image_row)
        counts = self._get_counts(per_image_row)
        predicted_class = self._get_predicted_class(per_image_row)

        row_dicts = []

        sampler = ConceptCountsSamplingIterator(self.id_samples_broadcast.value,
                                                self.counts_samples_broadcast.value)

        # each image has multiple pixel influence masks from different estimators
        for influence_estimator, influence_mask in zip(per_image_row[Field.INFLUENCE_ESTIMATOR.name],
                                                       per_image_row[Field.INFLUENCE_MASK.name]):
            influence_estimator = Field.INFLUENCE_ESTIMATOR.decode(influence_estimator)
            influence_mask = Field.INFLUENCE_MASK.decode(influence_mask)

            for detector in self.detectors:
                influential_counts = np.zeros((len(self.all_concept_names),), dtype=np.uint8)
                for concept_names, concept_masks in zip(per_image_row[Field.CONCEPT_NAMES.name],
                                                        map(Field.CONCEPT_MASKS.decode,
                                                            per_image_row[Field.CONCEPT_MASKS.name])):
                    for concept_name, concept_mask in zip(concept_names, concept_masks):
                        if detector.detect(influence_mask, concept_mask):
                            influential_counts[self.all_concept_names.index(concept_name)] += 1
                            self._log_item('{} {}: Concept {} has exceptional influence.'
                                           .format(influence_estimator, detector, concept_name))

                for perturber in self.perturbers:
                    for perturbed_counts, perturbed_image_id \
                            in perturber.perturb(influential_counts, counts, sampler):
                        row_dicts.append({Field.PREDICTED_CLASS.name: predicted_class,
                                          Field.INFLUENCE_ESTIMATOR.name: influence_estimator,
                                          Field.PERTURBER.name: str(perturber),
                                          Field.DETECTOR.name: str(detector),
                                          Field.IMAGE_ID.name: image_id,
                                          Field.PERTURBED_IMAGE_ID.name: perturbed_image_id,
                                          **dict(zip(self.all_concept_names, perturbed_counts))})

        return row_dicts

    def to_df(self) -> DataFrame:
        with self._log_task('Searching influential concepts on images:'):
            perturbed_rdd = self.union_df.join(self._get_influences_df(), on=Field.IMAGE_ID.name, how='inner').rdd \
                .flatMap(self._perturb_concepts_on_image) \
                .map(lambda r: dict_to_spark_row(self.output_schema, r))
        return self.spark_cfg.session.createDataFrame(perturbed_rdd, self.output_schema.as_spark_schema())


@dataclass
class CLInterface:
    # how to use spark
    spark_cfg: SparkSessionConfig

    # url of petastorm parquet store of schema `Schema.PIXEL_INFLUENCES`
    influences_url: str

    # urls of petastorm parquet stores of schema `Schema.CONCEPT_MASKS`
    concept_mask_urls: List[str]

    # which detectors to use
    detectors: List[InfluenceDetector] = field(default_factory=list, init=False)

    # which perturbers to use
    perturbers: List[Perturber] = field(default_factory=list, init=False)

    # thresholds to run detector with
    detection_thresholds: List[float]

    # how many perturbed object counts to generate per image using the `GlobalPerturber`
    global_perturbations_per_image: List[int]

    # how many perturbed object counts to generate per image using the `LocalPerturber`
    max_local_perturbations_per_image: List[int]

    def __post_init__(self):
        detectors = [LiftInfluenceDetector(threshold) for threshold in self.detection_thresholds]
        perturbers = [GlobalPerturber(num_perturbations)
                      for num_perturbations in self.global_perturbations_per_image]
        perturbers += [LocalPerturber(max_perturbations)
                       for max_perturbations in self.max_local_perturbations_per_image]
        self.generator = PerturbedConceptCountsGenerator(self.spark_cfg, self.concept_mask_urls,
                                                         self.influences_url, detectors, perturbers)


if __name__ == '__main__':
    parser = ArgumentParser(description='Leverage pixel observe_influence estimators, such as LIME or DeepLift,'
                                        'to improve the training data for surrogate models of an image classifier.'
                                        'The generated training data is not in the pixel space used by the classifier,'
                                        'but in the space of "concept-counts" on images.')
    parser.add_arguments(CLInterface, dest='cli')
    parser.add_arguments(LoggingConfig, dest='logging')
    parsed, remaining = parser.parse_known_args()
    generator: PerturbedConceptCountsGenerator = parsed.cli.generator

    parser = ArgumentParser()

    @dataclass
    class ConceptsWriteConfig(PetastormWriteConfig):
        output_schema: Unischema = field(default_factory=lambda: generator.output_schema, init=False)

    parser.add_arguments(ConceptsWriteConfig, dest='write_cfg')
    parsed = parser.parse_args(remaining)
    write_cfg: PetastormWriteConfig = parsed.write_cfg

    generator.spark_cfg.write_petastorm(generator.to_df(), write_cfg)
