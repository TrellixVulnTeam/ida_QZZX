import abc
import csv
import logging
import multiprocessing as mp
import os
import queue
import re
import time
from collections import Counter
from colorsys import rgb_to_hls
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce, partial
from pathlib import Path
from typing import Optional, Iterable, Type

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from simple_parsing import ArgumentParser
from torch.nn import DataParallel
from torchvision.transforms.functional import to_pil_image

from petastorm.codecs import ScalarCodec, CompressedImageCodec, \
    CompressedNdarrayCodec
from petastorm.etl.dataset_metadata import get_schema_from_dataset_url
from petastorm.unischema import UnischemaField, dict_to_spark_row, Unischema

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType, FloatType

from simadv.common import PetastormReadConfig, LoggingConfig, PetastormTransformer, TorchConfig, RowDict
from simadv.places365.hub import Places365Hub
from simadv.util.functools import cached_property
from simadv.util.webcache import WebCache


class DataGenerator(abc.ABC):
    """
    When called, this class generates a dataframe with a certain schema.
    """

    def __init__(self, id_field: UnischemaField):
        self.id_field = id_field
        self._log_nesting = 0

    @property
    @abc.abstractmethod
    def fields(self) -> [UnischemaField]:
        """
        List of all output fields, except the primary key field `self.id_field`.
        """
        pass

    @cached_property
    def schema(self):
        """
        Output schema of this describer as a petastorm `Unischema`.
        """
        return Unischema('DescriberSchema', [self.id_field] + self.fields)

    @abc.abstractmethod
    def __call__(self) -> DataFrame:
        """
        Generate a dataframe with a schema conforming to `self.schema`.
        """

    def _log(self, msg):
        prefix = '--' * self._log_nesting + ' ' if self._log_nesting else ''
        logging.info(prefix + msg)

    def _log_item(self, msg):
        self._log('<{}/>'.format(msg))

    def _log_group_start(self, msg):
        self._log('<{}>'.format(msg))
        self._log_nesting += 1

    def _log_group_end(self):
        self._log_nesting -= 1
        self._log('<done/>')

    @contextmanager
    def _log_task(self, msg):
        self._log_group_start(msg)
        try:
            yield None
        finally:
            self._log_group_end()


class CopyDataGenerator(DataGenerator):
    """
    Returns the input rows unchanged.
    """

    def __init__(self, spark_session: SparkSession, input_url: str,
                 id_field: UnischemaField):
        super().__init__(id_field)
        self.spark_session = spark_session
        self.input_url = input_url

        schema = get_schema_from_dataset_url(self.input_url)
        fields = schema.fields.values()
        assert id_field in fields
        self._fields = [f for f in fields if f != self.id_field]

    @property
    def fields(self) -> [UnischemaField]:
        return self._fields

    def __call__(self):
        with self._log_task('Describing with: unchanged input'):
            df = self.spark_session.read.parquet(self.input_url)
            self._log_item('Described a total of {} rows.'.format(df.count()))
            return df


class DictBasedDataGenerator(DataGenerator):

    def __init__(self, spark_session: SparkSession, output_description: str,
                 id_field: UnischemaField):
        super().__init__(id_field)
        self.output_description = output_description
        self.spark_session = spark_session

    @abc.abstractmethod
    def generate(self) -> Iterable[RowDict]:
        """
        Generate a sequence of rows with a schema conforming to `self.schema`.
        """
        pass

    def __call__(self) -> DataFrame:
        with self._log_task('Describing with: {}'
                            .format(self.output_description)):
            rows = []

            last_time = time.time()

            for num_rows, row_dict in enumerate(self.generate()):
                rows.append(dict_to_spark_row(self.schema, row_dict))

                current_time = time.time()
                if current_time - last_time > 5:
                    self._log_item('Described {} rows so far.'
                                   .format(num_rows + 1))
                    last_time = current_time

            spark_schema = self.schema.as_spark_schema()
            df = self.spark_session.createDataFrame(rows, spark_schema)
            return df


class TorchDescriber(DictBasedDataGenerator, abc.ABC):
    """
    Reads batches of image data from a petastorm store
    and converts these images to Torch tensors.

    Attention: If the input images can have different sizes, you *must*
    set `read_cfg.batch_size` to 1!

    Subclasses can describe the produced tensors as other data
    batch-wise by implementing the `describe_batch` method.
    """

    def __init__(self, spark_session: SparkSession,
                 output_description: str,
                 id_field: UnischemaField,
                 read_cfg: PetastormReadConfig,
                 device: torch.device):
        """
        Reads image data from a petastorm `read_cfg` and describes it as other data.
        The input petastorm schema must have a field called *image*.
        """
        super().__init__(spark_session, output_description, id_field)

        self.read_cfg = read_cfg
        self.device = device

    def batch_iter(self):
        def to_tensor(batch_columns):
            row_ids, image_arrays = batch_columns
            image_tensors = torch.as_tensor(image_arrays) \
                .to(self.device, torch.float) \
                .div(255)

            return row_ids, image_tensors

        current_batch = []
        reader = self.read_cfg.make_reader(['image', self.id_field.name])

        for row in reader:
            row_id = getattr(row, self.id_field.name)
            current_batch.append((row_id, row.image))

            if len(current_batch) < self.read_cfg.batch_size:
                continue

            yield to_tensor(list(zip(*current_batch)))
            current_batch.clear()

        if current_batch:
            yield to_tensor(list(zip(*current_batch)))

    @abc.abstractmethod
    def describe_batch(self, ids, batch) -> Iterable[RowDict]:
        pass

    def cleanup(self):
        """
        Can be overridden to clean up more stuff.
        """
        torch.cuda.empty_cache()  # release all memory that can be released

    def generate(self):
        with torch.no_grad():
            for ids, batch in self.batch_iter():
                yield from self.describe_batch(ids, batch)

        self.cleanup()


class ImageDummyDescriber(TorchDescriber):
    """
    Passes each image through, untouched.
    """

    def __init__(self,  spark_session: SparkSession,
                 id_field: UnischemaField,
                 read_cfg: PetastormReadConfig,
                 device: torch.device, width: int, height: int):
        super().__init__(spark_session,
                         'the untouched original image', id_field, read_cfg,
                         device)
        self._field = UnischemaField('image', np.uint8, (width, height, 3),
                                     CompressedImageCodec('png'), False)
        self._width = width
        self._height = height

    @property
    def fields(self) -> [UnischemaField]:
        return [self._field]

    def describe_batch(self, ids, image_tensors):
        image_tensors = image_tensors.cpu().permute(0, 3, 1, 2)

        for row_id, image_tensor in zip(ids, image_tensors):
            image_arr = np.asarray(to_pil_image(image_tensor, 'RGB')
                                   .resize((self._width, self._height)))
            yield {self.id_field.name: row_id,
                   self._field.name: image_arr}


class ColorDistributionDescriber(TorchDescriber):
    """
    Describes each image with a "color distribution".
    That is, every pixel of the image is sorted into a bin
    of a histogram.
    Each bin corresponds to a perceivable color.
    """

    #
    BIN_NAMES = ['red', 'orange', 'gold', 'yellow',
                 'green', 'turquoise', 'blue',
                 'purple', 'magenta',
                 'black', 'white', 'grey']

    # assign natural language color names to hue values
    HUE_MAP = {20.: 'red',
               45.: 'orange',
               55.: 'gold',
               65.: 'yellow',
               155.: 'green',
               185.: 'turquoise',
               250.: 'blue',
               280.: 'purple',
               320.: 'magenta',
               360.: 'red'}

    def __init__(self, spark_session: SparkSession,
                 id_field: UnischemaField, read_cfg: PetastormReadConfig,
                 device: torch.device, resize_to: int = 100, debug: bool = False):
        super().__init__(spark_session,
                         'distribution of colors in the image', id_field,
                         read_cfg, device)
        self._hue_bin_names = list(self.HUE_MAP.values())
        self._hue_names = set(self._hue_bin_names)
        self._lightness_names = {'black', 'white', 'grey'}
        self._hue_bins = np.array([0.] + list(self.HUE_MAP.keys())) \
            / 360. * 255.

        self._fields = [UnischemaField(self._get_field_name(name),
                                       float, (), ScalarCodec(FloatType()),
                                       False)
                        for name in self._hue_names | self._lightness_names]

        self._resize_to = resize_to
        self._pool = None

        self._debug = debug

    @staticmethod
    def _get_field_name(color_name):
        return 'fraction_of_{}_pixels'.format(color_name)

    @property
    def fields(self) -> [UnischemaField]:
        return self._fields

    def generate(self):
        num_processes = min(3, os.cpu_count())

        with mp.Pool(num_processes) as pool:
            self._pool = pool
            yield from super().generate()

    @staticmethod
    def _pixel_row_to_color_dist(px_row: np.ndarray, hue_bins):
        black, white, grey = 0., 0., 0.
        hues = []

        for px in px_row:
            r, g, b = px / 255.
            hue, lightness, saturation = rgb_to_hls(r, g, b)
            if lightness < .1:
                black += 1.
            elif lightness > .9:
                white += 1.
            elif saturation < .1:
                if lightness < .85:
                    grey += 1.
                else:
                    white += 1.
            elif saturation < .7:
                # color of pixel is undefined, not clear enough
                continue
            else:
                hues.append(min(255., round(hue * 255.)))

        non_hue_counts = np.array([black, white, grey])

        hues = np.asarray(hues)
        hue_counts, _ = np.histogram(hues, hue_bins, range(256))

        return np.concatenate((non_hue_counts, hue_counts))

    @staticmethod
    def _describe_single(image_tensor, resize_to, hue_bins, hue_names,
                         hue_bin_names, lightness_names):
        image = to_pil_image(image_tensor, 'RGB') \
            .resize((resize_to, resize_to))

        image_arr = np.asarray(image)
        f = partial(ColorDistributionDescriber._pixel_row_to_color_dist,
                    hue_bins=hue_bins)
        counts_arrays = np.asarray([f(px_row) for px_row in image_arr])

        binned_px_count = np.sum(counts_arrays)
        if binned_px_count == 0.:
            num_colors = len(hue_names) + len(lightness_names)
            uniform_fraction = 1. / num_colors

            color_fractions = {name: uniform_fraction
                               for name in hue_names | lightness_names}
        else:
            fractions_arr = np.sum(counts_arrays, 0) / binned_px_count

            black, white, grey = fractions_arr[:3]
            color_fractions = {'black': black,
                               'white': white,
                               'grey': grey}

            color_fractions.update({hue: .0 for hue in hue_names})
            hue_fractions = fractions_arr[3:]
            for hue_fraction, hue_bin in zip(hue_fractions, hue_bin_names):
                color_fractions[hue_bin] += hue_fraction

        return color_fractions

    def describe_batch(self, ids, image_tensors):
        image_tensors = image_tensors.cpu().permute(0, 3, 1, 2)

        f = partial(self._describe_single, resize_to=self._resize_to,
                    hue_bins=self._hue_bins, hue_names=self._hue_names,
                    hue_bin_names=self._hue_bin_names,
                    lightness_names=self._lightness_names)
        fractions_batch = list(self._pool.map(f, image_tensors))

        for img_no, row_id, color_fractions in zip(range(len(image_tensors)),
                                                   ids, fractions_batch):
            row_dict = {self._get_field_name(color): fraction
                        for color, fraction in color_fractions.items()}
            row_dict[self.id_field.name] = row_id

            if self._debug:
                self._log('image shows the following colors:\n{}'
                          .format(sorted(color_fractions.items(),
                                         key=lambda x: x[1],
                                         reverse=True)))
                to_pil_image(image_tensors[img_no], mode='RGB').show()
                input('--- press enter to continue ---')

            yield row_dict


def _wrap_parallel(model):
    if torch.cuda.device_count() > 1:
        return DataParallel(model)
    return model


class TorchModelDescriber(TorchDescriber, abc.ABC):
    """
    Loads a Torch ML model before consuming data.
    Useful to load the model first on the GPU before the data.
    """

    def __init__(self, spark_session: SparkSession,
                 output_description: str, id_field: UnischemaField,
                 read_cfg: PetastormReadConfig,
                 device: torch.device, create_model_func):
        super().__init__(spark_session, output_description, id_field,
                         read_cfg, device)
        self._create_model = create_model_func
        self.model = None

    def cleanup(self):
        del self.model
        super().cleanup()

    def __call__(self):
        with torch.no_grad():
            # allocate model before tensors
            self.model = self._create_model()
        return super().__call__()


class Places365SceneLabelDescriber(TorchModelDescriber):
    """
    Translates each image to one out of 365 natural language scene labels,
    from the Places365 Challenge.
    """

    def __init__(self, spark_session: SparkSession,
                 id_field: UnischemaField, read_cfg: PetastormReadConfig,
                 device: torch.device, create_model_func,
                 top_k: Optional[int] = None):
        super().__init__(spark_session,
                         'scene names from the Places365 set', id_field,
                         read_cfg, device, create_model_func)

        assert top_k is None or top_k > 0
        self._top_k = top_k

        if self._top_k is not None:
            self._fields = [UnischemaField(self._get_top_k_field_name(i),
                                           np.uint8, (),
                                           ScalarCodec(IntegerType()),
                                           False)
                            for i in range(top_k)]
        else:
            self._fields = [UnischemaField('places365_scene_probs',
                                           np.float32, (365,),
                                           CompressedNdarrayCodec(),
                                           False)]

    def _get_top_k_field_name(self, position):
        assert 0 <= position < self._top_k
        return 'places365_top_{}_scene_label_id'.format(position + 1)

    @property
    def fields(self):
        return self._fields

    @staticmethod
    def with_resnet18(spark_session: SparkSession,
                      id_field: UnischemaField,
                      read_cfg: PetastormReadConfig, device: torch.device,
                      cache: WebCache, top_k: Optional[int] = None):
        def create_model():
            hub = Places365Hub(cache)
            return _wrap_parallel(hub.resnet18().to(device).eval())
        return Places365SceneLabelDescriber(spark_session,
                                            id_field, read_cfg, device,
                                            create_model, top_k=top_k)

    def describe_batch(self, ids, image_tensors):
        image_tensors = image_tensors.permute(0, 3, 1, 2)

        probs_batch = self.model(image_tensors).cpu()

        if self._top_k is None:
            for row_id, probs in zip(ids, probs_batch):
                yield {self.id_field.name: row_id,
                       'places365_scene_probs': probs.numpy()}
        else:
            _, predicted_labels_batch = torch.topk(probs_batch, self._top_k, -1)
            for row_id, predicted_labels in zip(ids, predicted_labels_batch):
                row_dict = {f.name: predicted_labels[i].item()
                            for i, f in enumerate(self._fields)}
                row_dict[self.id_field.name] = row_id
                yield row_dict


class CocoObjectNamesDescriber(TorchModelDescriber):
    """
    Translates each image to counts of detected objects from the COCO task.
    The COCO task includes 91 objects.
    """

    _OBJECT_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter',
        'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'N/A',
        'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, spark_session: SparkSession,
                 id_field: UnischemaField, read_cfg: PetastormReadConfig,
                 device: torch.device, create_model_func, threshold: float,
                 debug: bool = False):
        super().__init__(spark_session,
                         'detected objects from the COCO set', id_field,
                         read_cfg, device, create_model_func)

        assert 0 <= threshold < 1
        self.threshold = threshold

        num_ints = len(self.object_names())
        self._field = UnischemaField('coco_object_counts', np.uint8,
                                     (num_ints,), CompressedNdarrayCodec(),
                                     False)

        self.debug = debug

    @property
    def fields(self):
        return [self._field]

    @staticmethod
    def object_names():
        return CocoObjectNamesDescriber._OBJECT_NAMES

    @staticmethod
    def with_faster_r_cnn(spark_session: SparkSession,
                          id_field: UnischemaField,
                          read_cfg: PetastormReadConfig,
                          device: torch.device, **kwargs):
        def create_model():
            return torchvision.models.detection\
                .fasterrcnn_resnet50_fpn(pretrained=True)\
                .to(device)\
                .eval()
        return CocoObjectNamesDescriber(spark_session, id_field,
                                        read_cfg, device, create_model,
                                        **kwargs)

    @staticmethod
    def get_object_names_counts(counts_array):
        """
        Returns a mapping from object names to counts,
        as encoded in *counts_array*.
        """
        return {CocoObjectNamesDescriber.object_names()[obj_id]: count
                for obj_id, count in enumerate(counts_array) if count > 0}

    def describe_batch(self, ids, image_tensors):
        # transform (H, W, C) images to (C, H, W) images
        image_tensors = image_tensors.permute(0, 3, 1, 2)

        for row_id, image_tensor, result \
                in zip(ids, image_tensors, self.model(image_tensors)):
            obj_ids, scores = result['labels'], result['scores']
            obj_counts = Counter(obj_id.cpu().item()
                                 for score, obj_id in zip(scores, obj_ids)
                                 if score > self.threshold)

            obj_id_gen = range(len(self.object_names()))
            counts_arr = np.array([obj_counts.get(obj_id, 0)
                                   for obj_id in obj_id_gen],
                                  dtype=np.uint8)

            if self.debug:
                boxes = result['boxes'].tolist()

                image = to_pil_image(image_tensor, 'RGB')
                draw = ImageDraw.Draw(image)
                for obj_id, box, score in zip(obj_ids, boxes, scores):
                    if score > self.threshold:
                        draw.rectangle(list(box))
                        draw.text(box[:2], self.object_names()[obj_id])

                image.show()

                input('--- press enter to continue ---')

            yield {self.id_field.name: row_id,
                   self._field.name: counts_arr}


class TFDescriber(DictBasedDataGenerator, abc.ABC):
    """
    Reads batches of image data from a petastorm store
    and converts these images to Tensorflow tensors.
    Subclasses can describe these tensors as other data.

    FIXME: this describer currently only works if all images have the same size
    """

    def __init__(self, spark_session: SparkSession,
                 output_description: str, id_field: UnischemaField,
                 read_cfg: PetastormReadConfig):
        super().__init__(spark_session, output_description, id_field)

        self.read_cfg = read_cfg

    def batch_iter(self):
        for schema_view in self.read_cfg.make_tf_dataset(['image', self.id_field.name]):
            ids = getattr(schema_view, self.id_field.name).numpy()\
                .astype(self.id_field.numpy_dtype).tolist()
            yield ids, schema_view.image


class TFObjectDetectionProcess(mp.Process):
    """
    Executes an object detection model in a separate process
    and delivers the results with a queue.
    """

    def __init__(self, model_name: str, cache: WebCache,
                 in_queue: mp.JoinableQueue, out_queue: mp.Queue,
                 gpu_id: Optional[int]):
        super().__init__()

        self.model_name = model_name
        self.cache = cache
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.gpu_id = gpu_id

    def run(self):
        if self.gpu_id is not None:
            # set GPU id before importing tensorflow
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(self.gpu_id)

        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # import tensorflow locally in the process
        import tensorflow as tf

        model_dir = self.cache.get_absolute_path(self.model_name) / 'saved_model'

        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']

        while True:
            ids, in_batch = self.in_queue.get()
            if in_batch is None:
                logging.info('Exiting Process for GPU {}'.format(self.gpu_id))
                self.in_queue.task_done()
                break

            out_batch = model(in_batch)
            self.out_queue.put((ids, in_batch, out_batch))
            self.in_queue.task_done()


class OIV4ObjectNamesDescriber(TFDescriber):
    """
    Translates each image to a set of detected objects from the OpenImages task.
    The OpenImages task includes 600 objects.
    """

    MODELS_URL = 'http://download.tensorflow.org/models/object_detection/'
    LABELS_URL = 'https://storage.googleapis.com/openimages/2018_04/'

    def __init__(self, spark_session: SparkSession,
                 id_field: UnischemaField,
                 read_cfg: PetastormReadConfig, model_name: str,
                 cache: WebCache, threshold: float, debug: bool = False):
        super().__init__(spark_session,
                         'detected objects from the OpenImages set', id_field,
                         read_cfg)

        cache.cache(model_name + '.tar.gz', self.MODELS_URL, is_archive=True)

        num_gpus = torch.cuda.device_count()
        gpus = range(num_gpus) if num_gpus > 0 else (None,)

        self.in_queue = mp.JoinableQueue(maxsize=max(1, num_gpus))
        self.out_queue = mp.Queue(maxsize=read_cfg.batch_size)
        self.processes = [TFObjectDetectionProcess(model_name, cache,
                                                   self.in_queue,
                                                   self.out_queue,
                                                   gpu)
                          for gpu in gpus]

        assert 0 <= threshold < 1
        self.threshold = threshold

        self.cache = cache

        num_ints = len(self.object_names())
        self._field = UnischemaField('oi_object_counts', np.uint8,
                                     (num_ints,), CompressedNdarrayCodec(),
                                     False)

        self.debug = debug

    def object_names(self):
        return self.object_names_from_cache(self.cache)

    @staticmethod
    def object_names_from_cache(cache: WebCache):
        url = OIV4ObjectNamesDescriber.LABELS_URL

        with cache.open('class-descriptions-boxable.csv', url)\
                as object_names_file:
            csv_reader = csv.reader(object_names_file, delimiter=',')
            return ['__background__'] + [row[1] for row in csv_reader]

    @property
    def fields(self) -> [UnischemaField]:
        return [self._field]

    @staticmethod
    def with_pretrained_model(spark_session: SparkSession,
                              model_name: str, id_field: UnischemaField,
                              read_cfg: PetastormReadConfig,
                              cache: WebCache, **kwargs):
        return OIV4ObjectNamesDescriber(spark_session, id_field,
                                        read_cfg, model_name, cache,
                                        **kwargs)

    def result_iter(self):
        for p in self.processes:
            p.start()

        pending_count = 0
        for ids, batch in self.batch_iter():
            self.in_queue.put((ids, batch))  # blocks until a slot is free
            pending_count += 1

            try:
                yield self.out_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                pending_count -= 1

        for _ in self.processes:
            self.in_queue.put((None, None))
        self.in_queue.join()  # wait until all results are there

        for _ in range(pending_count):
            yield self.out_queue.get()

        for p in self.processes:
            p.join()

    def generate(self):
        for ids, image_tensors, results in self.result_iter():
            obj_ids_batches = results['detection_classes'].numpy().astype(int)
            scores_batches = results['detection_scores'].numpy()
            boxes_batches = results['detection_boxes'].numpy()

            for row_id, image_tensor, obj_ids, scores, boxes in \
                    zip(ids, image_tensors, obj_ids_batches, scores_batches,
                        boxes_batches):
                obj_counts = Counter(obj_id for score, obj_id
                                     in zip(scores, obj_ids)
                                     if score > self.threshold)

                all_obj_ids = range(len(self.object_names()))
                counts_arr = np.array([obj_counts.get(obj_id, 0)
                                       for obj_id in all_obj_ids],
                                      dtype=np.uint8)

                if self.debug:
                    image = Image.fromarray(image_tensor.numpy(), 'RGB')
                    draw = ImageDraw.Draw(image)
                    for obj_id, box, score in zip(obj_ids, boxes, scores):
                        if score > self.threshold:
                            top, left = (box[0] * image.height,
                                         box[1] * image.width)
                            bottom, right = (box[2] * image.height,
                                             box[3] * image.width)
                            draw.rectangle((left, bottom, right, top),
                                           outline=(255, 0, 0))
                            draw.text((left, top), self.object_names()[obj_id],
                                      fill=(255, 0, 0))

                    image.show()

                    input('--- press enter to continue ---')

                yield {self.id_field.name: row_id,
                       self._field.name: counts_arr}


class JoinDataGenerator(DataGenerator):
    """
    Executes multiple data generators sequentially.
    Then joins the results based on their `id_field`.
    """

    def __init__(self, generators: [DataGenerator]):
        if len(set(g.id_field for g in generators)) > 1:
            raise ValueError('All generators must use the same ID field '
                             'for the join.')

        super().__init__(generators[0].id_field)
        self.generators = generators

    @property
    def fields(self) -> [UnischemaField]:
        return [f for g in self.generators for f in g.fields]

    def __call__(self):
        with self._log_task('Joining multiple data generators...'):
            dfs = [g() for g in self.generators]
            # use outer join in case there are missing values
            op = partial(DataFrame.join, on=self.id_field.name, how='outer')
            return reduce(op, dfs)


_RE_COPY = re.compile(r'copy')
_RE_IMAGE = re.compile(r'(?P<name>images)\[(?P<width>\d+),\s?(?P<height>\d+)]')
_RE_COLORS = re.compile(r'(?P<name>colors)(?P<debug>\+debug)?')
_RE_PLACES365 = re.compile(r'(?P<name>places365_scenes)\[(?P<model>.*)]'
                           r'(@(?P<k>\d+))?')
_RE_COCO = re.compile(r'(?P<name>coco_objects)'
                      r'\[(?P<model>[^+]+)(?P<debug>\+debug)?]'
                      r'(@(?P<t>0?\.\d+))?')
_RE_OI = re.compile(r'(?P<name>oi_objects)'
                    r'\[(?P<model>[^+]+)(?P<debug>\+debug)?]'
                    r'(@(?P<t>0?\.\d+))?')


@dataclass
class DescribeTask(PetastormTransformer):
    """
    Runs multiple describers with shared settings.
    """

    describers: str
    torch_cfg: TorchConfig
    cache_dir: str

    def __post_init__(self):
        self.cache = WebCache(Path(self.cache_dir))

    def _create(self, d_spec: str):
        """
        Creates a describer based on a parameter string `d_spec`.
        """

        def _raise_unknown_model(_model, _describer):
            raise ValueError('Unknown model {} for describer {}'
                             .format(_model, _describer))

        copy_match = _RE_COPY.fullmatch(d_spec)
        if copy_match:
            return CopyDataGenerator(self.write_cfg.session,
                                     self.read_cfg.input_url,
                                     self.id_field)

        image_match = _RE_IMAGE.fullmatch(d_spec)
        if image_match:
            d = image_match.groupdict()
            width, height = int(d['width']), int(d['height'])
            return ImageDummyDescriber(self.write_cfg.session,
                                       self.id_field, self.read_cfg,
                                       self.torch_cfg.device, width, height)

        color_match = _RE_COLORS.fullmatch(d_spec)
        if color_match:
            d = color_match.groupdict()
            params = {'debug': d['debug'] is not None}
            return ColorDistributionDescriber(self.write_cfg.session,
                                              self.id_field,
                                              self.read_cfg,
                                              self.torch_cfg.device, **params)

        places365_match = _RE_PLACES365.fullmatch(d_spec)
        if places365_match:
            d = places365_match.groupdict()
            model = d['model']
            params = {'top_k': int(d['k'])} if d['k'] else {}

            if model in ['default', 'resnet18']:
                return Places365SceneLabelDescriber \
                    .with_resnet18(self.write_cfg.session,
                                   self.id_field, self.read_cfg,
                                   self.torch_cfg.device, cache=self.cache,
                                   **params)

            _raise_unknown_model(model, d['name'])

        coco_match = _RE_COCO.fullmatch(d_spec)
        if coco_match:
            d = coco_match.groupdict()
            model = d['model']
            params = {'threshold': float(d['t'])} if d['t'] else {}
            params['debug'] = d['debug'] is not None

            if model in ['default', 'faster_r_cnn']:
                return CocoObjectNamesDescriber \
                    .with_faster_r_cnn(self.write_cfg.session,
                                       self.id_field, self.read_cfg,
                                       self.torch_cfg.device,
                                       **params)

            _raise_unknown_model(model, d['name'])

        oi_match = _RE_OI.fullmatch(d_spec)
        if oi_match:
            d = oi_match.groupdict()
            model_short = d['model']
            params = {'threshold': float(d['t'])} if d['t'] else {}
            params['debug'] = d['debug'] is not None

            model_full = {'inception_resnet': 'faster_rcnn_inception_resnet_v2'
                                              '_atrous_oid_v4_2018_12_12',
                          'mobilenet': 'ssd_mobilenet_v2_oid_v4_2018_12_12'}
            model_full['default'] = model_full['inception_resnet']

            try:
                model = model_full[model_short]
            except KeyError:
                _raise_unknown_model(model_short, d['name'])
            else:
                return OIV4ObjectNamesDescriber\
                    .with_pretrained_model(self.write_cfg.session, model,
                                           self.id_field, self.read_cfg,
                                           self.cache, **params)

        raise ValueError('Unknown describer: {}'.format(d_spec))

    def run(self):
        describers = [self._create(t_spec) for t_spec in self.describers]
        data_gen = JoinDataGenerator(describers)
        out_df = data_gen()

        logging.info('Writing dataframe of {} rows.'.format(out_df.count()))
        logging.info('First row: {}'.format(out_df.take(1)))

        self.write_cfg.write_parquet(out_df, data_gen.schema, self.output_url)

        logging.info('----- Finished -----')


def main(describe_task: Type[DescribeTask]):
    """
    Describe images with abstract and familiar attributes.
    The images are read from a petastorm parquet store. In this parquet store,
    there must be two fields:

      - A field holding a unique id for each image.
        This field may have any name, it is passed with the parameter `id_field`.
      - A field holding image data encoded with the petastorm png encoder.
        This field must be named *image*.

    Use one of the implementations in the submodules of this package.
    """
    mp.set_start_method('spawn')
    parser = ArgumentParser(description='Describe images with abstract and familiar attributes.')
    parser.add_arguments(describe_task, 'describe_task')
    parser.add_arguments(LoggingConfig, 'logging')
    args = parser.parse_args()
    args.describe_task.run()
