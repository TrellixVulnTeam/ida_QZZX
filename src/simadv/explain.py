import abc
import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from typing import Optional, Type

import numpy as np
from petastorm.codecs import ScalarCodec, CompressedNdarrayCodec
from petastorm.unischema import UnischemaField, Unischema, dict_to_spark_row
from pyspark.sql.types import StringType
from simple_parsing import ArgumentParser

from simadv.common import LoggingConfig, TorchConfig, PetastormTransformer, \
    TorchImageClassifier, TorchImageClassifierSerialization, Classifier, PetastormReadConfig

from anchor import anchor_image


class InfluenceEstimator(abc.ABC):
    """
    A method to estimate the most influential pixels for a given classifier prediction.
    """

    @property
    @abc.abstractmethod
    def id(self) -> str:
        pass

    @abc.abstractmethod
    def get_influential_pixels(self, classifier: Classifier, img: np.ndarray) -> np.ndarray:
        pass


@dataclass
class AnchorInfluenceEstimator(InfluenceEstimator):

    max_anchor_size = 2
    coverage_samples = 5
    stop_on_first = True
    tau = 0.2
    delta = 0.1
    batch_size = 10

    def __init__(self):
        self.anchor = anchor_image.AnchorImage()

    def id(self) -> str:
        return 'anchor'

    def get_influential_pixels(self, classifier: Classifier, img: np.ndarray) -> np.ndarray:
        explanation = self.anchor.explain_instance(np.asarray(img),
                                                   classifier.predict_proba,
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

        relevant_mask = np.zeros(segmentation_mask.shape, dtype=int)
        for segment_id, _, _, _, _ in relevant_segments:
            relevant_mask = np.bitwise_or(relevant_mask,
                                          segmentation_mask == segment_id)
        return relevant_mask


@dataclass
class TorchExplainTask(PetastormTransformer):
    classifier_serial: TorchImageClassifierSerialization
    torch_cfg: TorchConfig

    anchor_ie: AnchorInfluenceEstimator

    num_objects: int
    # the number of object classes annotated for the input images

    counts_field_name: Optional[str]
    # how to call the field holding the perturbed object counts

    class_field_name: Optional[str]
    # how to call the field holding the predicted class of the classifier

    estimator_field_name: Optional[str]
    # how to call the field holding the id of the used influence estimator

    original_image_id_field_name: str = field(default='original_image_id', init=False)

    perturbation_image_id_field_name: str = field(default='perturbation_image_id', init=False)

    min_coverage = 0.1
    # if more than this fraction of an objects bounding box is covered by influential pixels,
    # the object is considered relevant for the corresponding prediction of the classifier.

    perturbation_factor = 10
    # number of perturbed object counts to generate based on each influence mask

    def __post_init__(self):
        # load the classifier from its name
        @dataclass
        class PartialTorchImageClassifier(TorchImageClassifier):
            torch_cfg: TorchConfig = field(default_factory=lambda: self.torch_cfg, init=False)

        if self.counts_field_name is None:
            self.counts_field_name = 'object_counts'

        if self.class_field_name is None:
            self.class_field_name = self.classifier.name + '_prediction'

        if self.estimator_field_name is None:
            self.estimator_field_name = 'influence_estimator'

        self.classifier = PartialTorchImageClassifier.load(self.classifier_serial.path)
        self.influence_estimators = (self.anchor_ie,)
        self.sampling_read_cfg = PetastormReadConfig(self.read_cfg.input_url, self.read_cfg.batch_size, True,
                                                     self.read_cfg.pool_type, self.read_cfg.workers_count)

    @cached_property
    def schema(self):
        """
        Output schema of this task as a petastorm `Unischema`.
        """
        object_counts_field = UnischemaField(self.counts_field_name, np.uint8, (self.num_objects,),
                                             CompressedNdarrayCodec(), False)
        class_field = UnischemaField(self.class_field_name, np.float32, (self.classifier.task.num_classes,),
                                     CompressedNdarrayCodec(), False)  # probability distribution over classes
        estimator_field = UnischemaField('influence_estimator', np.unicode_, (), ScalarCodec(StringType()), False)
        original_image_id_field = UnischemaField(self.original_image_id_field_name, self.id_field.numpy_dtype,
                                                 self.id_field.shape, self.id_field.codec, self.id_field.nullable)
        perturbation_image_id_field = UnischemaField(self.perturbation_image_id_field_name,
                                                     self.id_field.numpy_dtype, self.id_field.shape,
                                                     self.id_field.codec, self.id_field.nullable)

        return Unischema('ExplainerSchema', [object_counts_field, class_field, estimator_field, original_image_id_field,
                                             perturbation_image_id_field])

    def _get_perturbations(self):
        last_time = time.time()
        processed_images_count = 0
        hit_counts = {est.id: 0 for est in self.influence_estimators}

        with self.read_cfg.make_reader([self.id_field.name, 'image', 'boxes']) as reader:
            with self.read_cfg.make_reader([self.id_field.name, 'boxes']) as sampling_reader:
                for row in reader:
                    row_id = getattr(row, self.id_field.name)
                    processed_images_count += 1

                    for influence_estimator in self.influence_estimators:
                        pred_class = np.argmax(self.classifier.predict_proba(np.expand_dims(row.image, 0))[0])
                        influence_mask = influence_estimator.get_influential_pixels(self.classifier, row.image)

                        counts = np.zeros((self.num_objects,))
                        min_counts = np.zeros((self.num_objects,))
                        for box in row.boxes:
                            obj_id, x_min, x_max, y_min, y_max = box[:4]
                            coverage = np.count_nonzero(influence_mask[x_min:x_max, y_min:y_max]) / \
                                       ((x_max - x_min) * (y_max - y_min))
                            counts[obj_id] += 1

                            if coverage > self.min_coverage:
                                min_counts[obj_id] += 1

                        has_influential_object = np.any(min_counts)

                        if has_influential_object:
                            # yield original counts
                            yield {self.counts_field_name: counts,
                                   self.class_field_name: pred_class,
                                   self.estimator_field_name: None,
                                   self.original_image_id_field_name: row_id,
                                   self.perturbation_image_id_field_name: None}

                            # sample random object counts from the same distribution
                            for sampled_row in islice(sampling_reader, self.perturbation_factor):
                                sample_counts = np.zeros((self.num_objects,))
                                for box in sampled_row.boxes:
                                    obj_id = box[0]
                                    sample_counts[obj_id] += 1

                                # keep all influential objects from the original image,
                                # change the rest based on the sample → pairwise maximum of counts
                                combined_counts = np.maximum(min_counts, sample_counts)

                                yield {self.counts_field_name: combined_counts,
                                       self.class_field_name: pred_class,
                                       self.estimator_field_name: influence_estimator.id,
                                       self.original_image_id_field_name: row_id,
                                       self.perturbation_image_id_field_name: getattr(sampled_row, self.id_field.name)}

                            hit_counts[influence_estimator.id] += 1

                    current_time = time.time()
                    if current_time - last_time > 5:
                        hit_freq = {hit_counts[est] / processed_images_count for est in self.influence_estimators}
                        logging.info('Processed {} images so far. Hit frequencies: {:.2f}'
                                     .format(processed_images_count, hit_freq))
                        last_time = current_time

    def run(self):
        spark_schema = self.schema.as_spark_schema()
        rows = [dict_to_spark_row(self.schema, row_dict) for row_dict in self._get_perturbations()]
        df = self.write_cfg.session.createDataFrame(rows, spark_schema)
        self.write_cfg.write_parquet(df, self.schema, self.output_url)


def main(explain_task: Type[TorchExplainTask]):
    """
    Explain an image classifier by representing its decision boundary with visible objects.
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
    parser = ArgumentParser(description='Explain an image classifier by representing its decision boundary '
                                        'with visible objects.')
    parser.add_arguments(explain_task, dest='explainer')
    parser.add_arguments(LoggingConfig, dest='logging')
    args = parser.parse_args()
    args.explainer.run()
