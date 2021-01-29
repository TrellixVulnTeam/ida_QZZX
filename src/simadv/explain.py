import abc
import logging
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients, Saliency, DeepLift, visualization as viz, GradientAttribution
from lime import lime_image
from matplotlib.patches import Rectangle
from petastorm.reader import Reader

from simadv.util.functools import cached_property
import itertools as it
from typing import Optional, Type, Any, Tuple, Iterable

import numpy as np
from petastorm.codecs import ScalarCodec, CompressedNdarrayCodec
from petastorm.unischema import UnischemaField, Unischema, dict_to_spark_row
from pyspark.sql.types import StringType, IntegerType
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
    def id(self) -> str:
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
    def id(self) -> str:
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
    def id(self) -> str:
        return 'igrad'


@dataclass
class SaliencyInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=Saliency, init=False)

    @property
    def id(self) -> str:
        return 'saliency'


@dataclass
class DeepLiftInfluenceEstimator(CaptumInfluenceEstimator):
    algorithm: GradientAttribution = field(default=DeepLift, init=False)

    @property
    def id(self) -> str:
        return 'deeplift'


class Perturber(abc.ABC):

    @property
    @abc.abstractmethod
    def id(self) -> str:
        pass

    @abc.abstractmethod
    def perturb(self, influential_counts: np.ndarray, counts: np.ndarray, sampler: Iterable[Tuple[np.ndarray, Any]]) \
            -> Tuple[np.ndarray, Any]:
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
        pass


@dataclass
class LocalPerturber(Perturber):
    """
    Assumes that given "influential objects" are a locally sufficient condition for the classification, i.e.,
    for images that are similar to the classified image.
    Hence drops all other, "non-influential" objects on this one image randomly -- they are noise.
    """

    @property
    def id(self) -> str:
        return 'local'

    def perturb(self, influential_counts: np.ndarray, counts: np.ndarray, sampler: Iterable[Tuple[np.ndarray, Any]]) \
            -> Tuple[np.ndarray, Any]:
        droppable_counts = counts - influential_counts
        gens = []
        for droppable_index in np.flatnonzero(droppable_counts):
            gens.append(zip(range(0, droppable_counts[droppable_index] + 1), it.repeat(droppable_index)))

        for drops in it.product(*gens):
            perturbed = counts.copy()
            for drop_count, drop_index in drops:
                perturbed[drop_index] -= drop_count
            yield perturbed, None


@dataclass
class GlobalPerturber(Perturber):
    """
    Assumes that given "influential objects" are a globally sufficient condition for the classification.
    Hence replaces all other objects randomly and assumes that the classification stays the same.
    """

    num_samples: int = 10

    @property
    def id(self) -> str:
        return 'global'

    def perturb(self, influential_counts: np.ndarray, counts: np.ndarray, sampler: Iterable[Tuple[np.ndarray, Any]]) \
            -> Tuple[np.ndarray, Any]:
        """
        Returns the original counts array plus `num_samples` (class parameter) additional arrays.
        The latter are unions of `influential_counts` with random counts drawn from `sampler`.
        """
        # original counts
        yield counts, None

        # sample random object counts from the same distribution
        for sample_counts, sample_id in it.islice(sampler, self.num_samples):
            # keep all influential objects from the original image,
            # change the rest based on the sample → pairwise maximum of counts
            combined_counts = np.maximum(influential_counts, sample_counts)
            yield combined_counts, sample_id


@dataclass
class TorchExplainTask(PetastormTransformer):
    classifier_serial: TorchImageClassifierSerialization
    torch_cfg: TorchConfig

    lime_ie: LIMEInfluenceEstimator
    anchor_ie: AnchorInfluenceEstimator
    igrad_ie: IntegratedGradientsInfluenceEstimator
    saliency_ie: SaliencyInfluenceEstimator

    drop_perturber: LocalPerturber
    sampling_perturber: GlobalPerturber

    # the number of object classes annotated for the input images
    num_objects: int

    # how to call the field holding the perturbed object counts
    counts_field_name: Optional[str]

    # how to call the field holding the predicted class of the classifier
    class_field_name: Optional[str]

    # how to call the field holding the id of the used influence estimator
    estimator_field_name: Optional[str]

    # how to call the field holding the id of the used perturber
    perturber_field_name: Optional[str]

    original_image_id_field_name: str = field(default='original_image_id', init=False)

    perturbation_image_id_field_name: str = field(default='perturbation_image_id', init=False)

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
        self.influence_estimators = (self.lime_ie, self.saliency_ie, self.igrad_ie)
        self.sampling_read_cfg = PetastormReadConfig(self.read_cfg.input_url, self.read_cfg.batch_size, True,
                                                     self.read_cfg.pool_type, self.read_cfg.workers_count)
        self.perturbers = (self.drop_perturber, self.sampling_perturber)

        if self.counts_field_name is None:
            self.counts_field_name = 'object_counts'

        if self.class_field_name is None:
            self.class_field_name = self.classifier.name + '_prediction'

        if self.estimator_field_name is None:
            self.estimator_field_name = 'influence_estimator'

        if self.perturber_field_name is None:
            self.perturber_field_name = 'perturber'

    @cached_property
    def schema(self):
        """
        Output schema of this task as a petastorm `Unischema`.
        """
        object_counts_field = UnischemaField(self.counts_field_name, np.uint8, (self.num_objects,),
                                             CompressedNdarrayCodec(), False)
        class_field = UnischemaField(self.class_field_name, np.uint16, (),
                                     ScalarCodec(IntegerType()), False)
        estimator_field = UnischemaField(self.estimator_field_name, np.unicode_, (), ScalarCodec(StringType()), True)
        perturber_field = UnischemaField(self.perturber_field_name, np.unicode_, (), ScalarCodec(StringType()), True)
        original_image_id_field = UnischemaField(self.original_image_id_field_name, self.id_field.numpy_dtype,
                                                 self.id_field.shape, self.id_field.codec, False)
        perturbation_image_id_field = UnischemaField(self.perturbation_image_id_field_name,
                                                     self.id_field.numpy_dtype, self.id_field.shape,
                                                     self.id_field.codec, True)

        return Unischema('ExplainerSchema', [object_counts_field, class_field, estimator_field, perturber_field,
                                             original_image_id_field, perturbation_image_id_field])

    def sampler(self, sampling_reader: Reader):
        for sampled_row in sampling_reader:
            sample_counts = np.zeros((self.num_objects,), dtype=np.uint8)
            for box in sampled_row.boxes:
                obj_id = box[0]
                sample_counts[obj_id] += 1
            yield sample_counts, getattr(sampled_row, self.id_field.name)

    def _get_perturbations(self):
        last_time = start_time = time.time()
        total_count = 0
        hit_counts = {est.id: 0 for est in self.influence_estimators}

        count_per_class = np.zeros((self.classifier.task.num_classes,))

        with self.read_cfg.make_reader([self.id_field.name, 'image', 'boxes']) as reader:
            with self.read_cfg.make_reader([self.id_field.name, 'boxes'], num_epochs=None) as sampling_reader:
                logging.info('Start reading images...')
                for row in reader:
                    pred = np.uint16(np.argmax(self.classifier.predict_proba(np.expand_dims(row.image, 0))[0]))

                    if self.observations_per_class is not None:
                        if count_per_class[pred] >= self.observations_per_class:
                            logging.info('Skipping, we already have enough observations of class {}.'.format(pred))
                            continue
                        elif np.sum(count_per_class) >= self.classifier.task.num_classes * self.observations_per_class:
                            break

                    count_per_class[pred] += 1

                    counts = np.zeros((self.num_objects,), dtype=np.uint8)
                    for box in row.boxes:
                        counts[box[0]] += 1

                    # yield the original observation
                    yield {self.counts_field_name: counts,
                           self.class_field_name: pred,
                           self.estimator_field_name: None,
                           self.perturber_field_name: None,
                           self.original_image_id_field_name: getattr(row, self.id_field.name),
                           self.perturbation_image_id_field_name: None}

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

                        height, width = influence_mask.shape
                        img_area = float(height * width)

                        min_counts = np.zeros((self.num_objects,), dtype=np.uint8)
                        for box in row.boxes:
                            obj_id, x_min, x_max, y_min, y_max = list(box)[:5]
                            x_min = int(np.floor(x_min * width))  # in case of "rounding doubt" let the object be larger
                            x_max = int(np.ceil(x_max * width))
                            y_min = int(np.floor(y_min * height))
                            y_max = int(np.ceil(y_max * height))
                            box_area = float((y_max - y_min) * (x_max - x_min))
                            # lift = how much more influence than expected do pixels of the box have?
                            lift = np.sum(influence_mask[y_min:y_max, x_min:x_max]) / (box_area / img_area)

                            logging.info('Influence of object {} is {:.2f} times the expected value.'
                                         .format(obj_id, lift))

                            if lift > self.lift_threshold:
                                min_counts[obj_id] += 1
                                logging.info('Object {} has exceptional influence: {:.2f} times the expected value.'
                                             .format(obj_id, lift))

                            if self.debug:
                                ax.add_patch(Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min),
                                                       linewidth=1, edgecolor='r', facecolor='none'))
                                ax.text(x_min, y_max, str(obj_id), color='r')

                        has_influential_object = np.any(min_counts)

                        if self.debug:
                            plt.show()
                            input('--- press enter to continue ---')
                            plt.clf()

                        if has_influential_object:
                            for perturber in self.perturbers:
                                for perturbed_count, image_id in perturber.perturb(min_counts, counts,
                                                                                   self.sampler(sampling_reader)):
                                    yield {self.counts_field_name: perturbed_count,
                                           self.class_field_name: pred,
                                           self.estimator_field_name: influence_estimator.id,
                                           self.perturber_field_name: perturber.id,
                                           self.original_image_id_field_name: getattr(row, self.id_field.name),
                                           self.perturbation_image_id_field_name: image_id}

                            hit_counts[influence_estimator.id] += 1

                    current_time = time.time()
                    if current_time - last_time > 5:
                        f = ', '.join(['{}: {:.2f}'.format(est.id, float(hit_counts[est.id]) / total_count)
                                      for est in self.influence_estimators])
                        logging.info('Processed {} images so far. Hit frequencies: {}'
                                     .format(total_count, f))
                        last_time = current_time

                    if self.time_limit_s is not None and current_time - start_time > self.time_limit_s:
                        logging.info('Reached timeout! Stopping.')
                        break

    def run(self):
        spark_schema = self.schema.as_spark_schema()
        rows = [dict_to_spark_row(self.schema, row_dict) for row_dict in self._get_perturbations()]
        df = self.write_cfg.session.createDataFrame(rows, spark_schema)
        logging.info('Writing {} object count observations to petastorm parquet store.'.format(df.count()))
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
