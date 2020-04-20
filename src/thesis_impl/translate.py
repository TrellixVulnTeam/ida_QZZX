import abc
import argparse
import csv
import logging
import multiprocessing as mp
import os
import queue
import re
import time
from collections import Counter
from contextlib import contextmanager
from functools import reduce, partial
from typing import Optional, Any, Dict, Iterable

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.nn import DataParallel
from torchvision.transforms.functional import to_pil_image

from petastorm import make_reader
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset, get_schema_from_dataset_url
from petastorm.tf_utils import make_petastorm_dataset
from petastorm.unischema import UnischemaField, dict_to_spark_row, Unischema

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType, FloatType

from thesis_impl.places365.hub import Places365Hub
from thesis_impl.io import torch_peta_loader
import thesis_impl.config as cfg
from thesis_impl.util.functools import cached_property
from thesis_impl.util.webcache import WebCache


RowDict = Dict[str, Any]


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
        Output schema of this translator as a petastorm `Unischema`.
        """
        return Unischema('TranslatorSchema', [self.id_field] + self.fields)

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
        super(CopyDataGenerator, self).__init__(id_field)
        self.spark_session = spark_session
        self.input_url = input_url
        self._schema = get_schema_from_dataset_url(self.input_url)

        assert id_field in self._schema.fields.values()

    @property
    def fields(self) -> [UnischemaField]:
        return self._schema.fields

    def __call__(self):
        with self._log_task('Translating to: unchanged input'):
            return self.spark_session.read.parquet(self.input_url)


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
        with self._log_task('Translating to: {}'
                            .format(self.output_description)):
            rows = []

            last_time = time.time()

            for num_rows, row_dict in enumerate(self.generate()):
                rows.append(dict_to_spark_row(self.schema, row_dict))

                current_time = time.time()
                if current_time - last_time > 5:
                    self._log_item('Translated {} rows so far.'
                                   .format(num_rows + 1))
                    last_time = current_time

            spark_schema = self.schema.as_spark_schema()
            df = self.spark_session.createDataFrame(rows, spark_schema)
            return df


class TorchTranslator(DictBasedDataGenerator, abc.ABC):
    """
    Reads batches of image data from a petastorm store
    and converts these images to Torch tensors.

    Subclasses can translate the produced tensors to other data
    batch-wise by implementing the `translate_batch` method.
    """

    def __init__(self, spark_session: SparkSession,
                 input_url: str, output_description: str,
                 id_field: UnischemaField,
                 read_cfg: cfg.PetastormReadConfig,
                 device: torch.device):
        """
        Reads image data from a petastorm `input_url` and translates it
        to other data.
        The input petastorm schema must have a field called *image*.
        """
        super().__init__(spark_session, output_description, id_field)

        self.input_url = input_url
        self.read_cfg = read_cfg
        self.device = device

    def batch_iter(self):
        for peta_batch in torch_peta_loader(self.input_url,
                                            self.read_cfg):
            images = peta_batch['image']\
                .to(self.device, torch.float)\
                .div(255)

            yield (peta_batch[self.id_field.name], images)

    @abc.abstractmethod
    def translate_batch(self, ids, batch) -> Iterable[RowDict]:
        pass

    def cleanup(self):
        """
        Can be overridden to clean up more stuff.
        """
        torch.cuda.empty_cache()  # release all memory that can be released

    def generate(self):
        with torch.no_grad():
            for ids, batch in self.batch_iter():
                yield from self.translate_batch(ids, batch)

        self.cleanup()


class ToImageDummyTranslator(TorchTranslator):
    """
    Passes each image through, untouched.
    """

    def __init__(self,  spark_session: SparkSession, input_url: str,
                 id_field: UnischemaField,
                 read_cfg: cfg.PetastormReadConfig,
                 device: torch.device, width: int, height: int):
        super().__init__(spark_session, input_url,
                         'the untouched original image', id_field, read_cfg,
                         device)
        self._field = UnischemaField('image', np.uint8, (width, height, 3),
                                     CompressedImageCodec('png'), False)
        self._width = width
        self._height = height

    @property
    def fields(self) -> [UnischemaField]:
        return [self._field]

    def translate_batch(self, ids, image_tensors):
        image_tensors = image_tensors.cpu().permute(0, 3, 1, 2)

        for row_id, image_tensor in zip(ids, image_tensors):
            image_arr = np.asarray(to_pil_image(image_tensor, 'RGB')
                                   .resize((self._width, self._height)))
            yield {self.id_field.name: row_id,
                   self._field.name: image_arr}


class ToColorDistributionTranslator(TorchTranslator):
    """
    Translates each image to a "hue distribution".
    That is, the distribution of the hue values of all pixels
    is estimated with a histogram.
    The intervals of this histogram correspond to natural language
    color names.
    """

    # assign natural language color names to hue values
    COLOR_MAP = {20.: 'red',
                 45.: 'orange',
                 55.: 'gold',
                 65.: 'yellow',
                 155.: 'green',
                 185.: 'turquoise',
                 250.: 'blue',
                 280.: 'purple',
                 320.: 'magenta',
                 360.: 'red'}

    def __init__(self, spark_session: SparkSession, input_url: str,
                 id_field: UnischemaField, read_cfg: cfg.PetastormReadConfig,
                 device: torch.device, num_colors: int=10, resize_to: int=100):
        super().__init__(spark_session, input_url,
                         'distribution of colors in the image', id_field,
                         read_cfg, device)
        self.num_colors = num_colors

        self._color_bins = self.COLOR_MAP.values()
        self._colors = set(self._color_bins)
        self._hue_bins = np.array([0.] + list(self.COLOR_MAP.keys())) \
            / 360. * 255.

        self._fields = [UnischemaField(self._get_field_name(color_name),
                                       float, (), ScalarCodec(FloatType()),
                                       False)
                        for color_name in self._colors]

        self._resize_to = resize_to

    def _get_field_name(self, color_name):
        return 'fraction_of_{}_pixels'.format(color_name)

    @property
    def fields(self) -> [UnischemaField]:
        return self._fields

    def translate_batch(self, ids, image_tensors):
        image_tensors = image_tensors.cpu().permute(0, 3, 1, 2)

        for row_id, image_tensor in zip(ids, image_tensors):
            image = to_pil_image(image_tensor, 'RGB')\
                .resize((self._resize_to, self._resize_to))\
                .convert('HSV')

            image_hues = np.asarray(image)[:, :, 0].flatten()
            hue_fractions, _ = np.histogram(image_hues, self._hue_bins,
                                            range(256))
            # we want a fraction of the total pixels
            hue_fractions = hue_fractions / (self._resize_to ** 2)

            color_fractions = {color: .0 for color in self._colors}

            for hue_fraction, color in zip(hue_fractions, self._color_bins):
                color_fractions[color] += hue_fraction

            row_dict = {self._get_field_name(color): fraction
                        for color, fraction in color_fractions.items()}
            row_dict[self.id_field.name] = row_id
            yield row_dict


def _wrap_parallel(model):
    if torch.cuda.device_count() > 1:
        return DataParallel(model)
    return model


class TorchModelTranslator(TorchTranslator, abc.ABC):
    """
    Loads a Torch ML model before consuming data.
    Useful to load the model first on the GPU before the data.
    """

    def __init__(self, spark_session: SparkSession, input_url: str,
                 output_description: str, id_field: UnischemaField,
                 read_cfg: cfg.PetastormReadConfig,
                 device: torch.device, create_model_func):
        super().__init__(spark_session, input_url, output_description, id_field,
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


class ToPlaces365SceneLabelTranslator(TorchModelTranslator):
    """
    Translates each image to one out of 365 natural language scene labels,
    from the Places365 Challenge.
    """

    def __init__(self, spark_session: SparkSession, input_url: str,
                 id_field: UnischemaField, read_cfg: cfg.PetastormReadConfig,
                 device: torch.device, create_model_func, top_k: int=1):
        super().__init__(spark_session, input_url,
                         'scene names from the Places365 set', id_field,
                         read_cfg, device, create_model_func)

        self._top_k = top_k
        self._fields = [UnischemaField(self._get_field_name(i),
                                       np.uint8, (),
                                       ScalarCodec(IntegerType()),
                                       False)
                        for i in range(top_k)]

    def _get_field_name(self, position):
        assert 0 <= position < self._top_k
        return 'places365_top_{}_scene_label_id'.format(position + 1)

    @property
    def fields(self):
        return self._fields

    @staticmethod
    def with_resnet18(spark_session: SparkSession, input_url,
                      id_field: UnischemaField,
                      read_cfg: cfg.PetastormReadConfig, device: torch.device,
                      cache: WebCache, top_k: int=5):
        def create_model():
            hub = Places365Hub(cache)
            return _wrap_parallel(hub.resnet18().to(device).eval())
        return ToPlaces365SceneLabelTranslator(spark_session, input_url,
                                               id_field, read_cfg, device,
                                               create_model, top_k=top_k)

    def translate_batch(self, ids, image_tensors):
        image_tensors = image_tensors.permute(0, 3, 1, 2)

        probs = self.model(image_tensors)
        _, predicted_labels_batch = torch.topk(probs, self._top_k, -1)
        for row_id, predicted_labels in zip(ids, predicted_labels_batch):
            row_dict = {f.name: predicted_labels[i].cpu().item()
                        for i, f in enumerate(self._fields)}
            row_dict[self.id_field.name] = row_id
            yield row_dict


class ToCocoObjectNamesTranslator(TorchModelTranslator):
    """
    Translates each image to a set of detected objects from the COCO task.
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

    def __init__(self, spark_session: SparkSession, input_url: str,
                 id_field: UnischemaField, read_cfg: cfg.PetastormReadConfig,
                 device: torch.device, create_model_func, threshold: float,
                 debug: bool=False):
        super().__init__(spark_session, input_url,
                         'detected objects from the COCO set', id_field,
                         read_cfg, device, create_model_func)

        assert 0 <= threshold < 1
        self.threshold = threshold

        num_ints = len(self.object_names())
        self._field = UnischemaField('coco_object_counts', np.uint8,
                                     (num_ints,), NdarrayCodec(), False)

        self.debug = debug

    @property
    def fields(self):
        return [self._field]

    @staticmethod
    def object_names():
        return ToCocoObjectNamesTranslator._OBJECT_NAMES

    @staticmethod
    def with_faster_r_cnn(spark_session: SparkSession, input_url: str,
                          id_field: UnischemaField,
                          read_cfg: cfg.PetastormReadConfig,
                          device: torch.device, **kwargs):
        def create_model():
            return torchvision.models.detection\
                .fasterrcnn_resnet50_fpn(pretrained=True)\
                .to(device)\
                .eval()
        return ToCocoObjectNamesTranslator(spark_session, input_url, id_field,
                                           read_cfg, device, create_model,
                                           **kwargs)

    @staticmethod
    def get_object_names_counts(counts_array):
        """
        Returns a mapping from object names to counts,
        as encoded in *counts_array*.
        """
        return {ToCocoObjectNamesTranslator.object_names()[obj_id]: count
                for obj_id, count in enumerate(counts_array) if count > 0}

    def translate_batch(self, ids, image_tensors):
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


class TFTranslator(DictBasedDataGenerator, abc.ABC):
    """
    Reads batches of image data from a petastorm store
    and converts these images to Tensorflow tensors.
    Subclasses can translate these tensors to other data.
    """

    def __init__(self, spark_session: SparkSession, input_url: str,
                 output_description: str, id_field: UnischemaField,
                 read_cfg: cfg.PetastormReadConfig):
        super().__init__(spark_session, output_description, id_field)

        self.input_url = input_url
        self.read_cfg = read_cfg

    def batch_iter(self):
        # we must set *shuffle_row_groups* to False, otherwise rows cannot
        # be joined with rows from other translators!
        reader = make_reader(self.input_url,
                             schema_fields=['image', self.id_field.name],
                             shuffle_row_groups=False)
        peta_dataset = make_petastorm_dataset(reader)\
            .batch(self.read_cfg.batch_size)

        for schema_view in peta_dataset:
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

        model_dir = self.cache.get_absolute_path(self.model_name) \
                    / 'saved_model'

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


class ToOIV4ObjectNamesTranslator(TFTranslator):
    """
    Translates each image to a set of detected objects from the OpenImages task.
    The OpenImages task includes 600 objects.
    """

    MODELS_URL = 'http://download.tensorflow.org/models/object_detection/'
    LABELS_URL = 'https://storage.googleapis.com/openimages/2018_04/'

    def __init__(self, spark_session: SparkSession, input_url: str,
                 id_field: UnischemaField,
                 read_cfg: cfg.PetastormReadConfig, model_name: str,
                 cache: WebCache, threshold: float, debug: bool=False):
        super().__init__(spark_session, input_url,
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
                                     (num_ints,), NdarrayCodec(), False)

        self.debug = debug

    def object_names(self):
        return self.object_names_from_cache(self.cache)

    @staticmethod
    def object_names_from_cache(cache: WebCache):
        url = ToOIV4ObjectNamesTranslator.LABELS_URL

        with cache.open('class-descriptions-boxable.csv', url)\
                as object_names_file:
            csv_reader = csv.reader(object_names_file, delimiter=',')
            return ['__background__'] + [row[1] for row in csv_reader]

    @property
    def fields(self) -> [UnischemaField]:
        return [self._field]

    @staticmethod
    def with_pretrained_model(spark_session: SparkSession,
                              model_name: str, input_url: str,
                              id_field: UnischemaField,
                              read_cfg: cfg.PetastormReadConfig,
                              cache: WebCache, **kwargs):
        return ToOIV4ObjectNamesTranslator(spark_session, input_url, id_field,
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
_RE_IMAGE = re.compile(r'(?P<name>images)\[(?P<width>\d+),\s?(?P<height>\d+)\]')
_RE_COLORS = re.compile(r'(?P<name>colors)(@(?P<k>\d+))?')
_RE_PLACES365 = re.compile(r'(?P<name>places365_scenes)\[(?P<model>.*)\]'
                           r'(@(?P<k>\d+))?')
_RE_COCO = re.compile(r'(?P<name>coco_objects)'
                      r'\[(?P<model>[^\+]+)(?P<debug>\+debug)?\]'
                      r'(@(?P<t>0?\.\d+))?')
_RE_OI = re.compile(r'(?P<name>oi_objects)'
                    r'\[(?P<model>[^\+]+)(?P<debug>\+debug)?\]'
                    r'(@(?P<t>0?\.\d+))?')


class TranslatorFactory:
    """
    Enables to create many translators with shared settings.
    """

    def __init__(self, input_url: str, id_field: UnischemaField,
                 read_cfg: cfg.PetastormReadConfig, torch_cfg: cfg.TorchConfig,
                 write_cfg: cfg.PetastormWriteConfig,
                 cache_dir: str):

        self.spark_session = SparkSession.builder \
            .config('spark.driver.memory',
                    write_cfg.spark_driver_memory) \
            .config('spark.executor.memory',
                    write_cfg.spark_exec_memory) \
            .master(write_cfg.spark_master) \
            .getOrCreate()

        self.input_url = input_url
        self.id_field = id_field
        self.read_cfg = read_cfg
        self.torch_device = torch_cfg.device
        self.cache = WebCache(cache_dir)

    def create(self, t_spec: str):
        """
        Creates a translator based on a parameter string `t_spec`.
        """

        def _raise_unknown_model(_model, _translator):
            raise ValueError('Unknown model {} for translator {}'
                             .format(_model, _translator))

        copy_match = _RE_COPY.fullmatch(t_spec)
        if copy_match:
            return CopyDataGenerator(self.spark_session, self.input_url,
                                     self.id_field)

        image_match = _RE_IMAGE.fullmatch(t_spec)
        if image_match:
            d = image_match.groupdict()
            width, height = int(d['width']), int(d['height'])
            return ToImageDummyTranslator(self.spark_session, self.input_url,
                                          self.id_field, self.read_cfg,
                                          self.torch_device, width, height)

        color_match = _RE_COLORS.fullmatch(t_spec)
        if color_match:
            d = color_match.groupdict()
            params = {'num_colors': int(d['k'])} if d['k'] else {}
            return ToColorDistributionTranslator(self.spark_session,
                                                 self.input_url, self.id_field,
                                                 self.read_cfg,
                                                 self.torch_device, **params)

        places365_match = _RE_PLACES365.fullmatch(t_spec)
        if places365_match:
            d = places365_match.groupdict()
            model = d['model']
            params = {'top_k': int(d['k'])} if d['k'] else {}

            if model in ['default', 'resnet18']:
                return ToPlaces365SceneLabelTranslator \
                    .with_resnet18(self.spark_session, self.input_url,
                                   self.id_field, self.read_cfg,
                                   self.torch_device, cache=self.cache,
                                   **params)

            _raise_unknown_model(model, d['name'])

        coco_match = _RE_COCO.fullmatch(t_spec)
        if coco_match:
            d = coco_match.groupdict()
            model = d['model']
            params = {'threshold': float(d['t'])} if d['t'] else {}
            params['debug'] = d['debug'] is not None

            if model in ['default', 'faster_r_cnn']:
                return ToCocoObjectNamesTranslator \
                    .with_faster_r_cnn(self.spark_session, self.input_url,
                                       self.id_field, self.read_cfg,
                                       self.torch_device,
                                       **params)

            _raise_unknown_model(model, d['name'])

        oi_match = _RE_OI.fullmatch(t_spec)
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
                return ToOIV4ObjectNamesTranslator\
                    .with_pretrained_model(self.spark_session, model,
                                           self.input_url, self.id_field,
                                           self.read_cfg,
                                           self.cache, **params)

        raise ValueError('Unknown translator: {}'.format(t_spec))


def _parse_args():
    parser = argparse.ArgumentParser(description='Translate images to low '
                                                 'dimensional and '
                                                 'interpretable kinds of data.')
    parser.add_argument('-i', '--input_url', type=str, required=True,
                        help='input URL for the images')
    parser.add_argument('-o', '--output_url', type=str, required=True,
                        help='output URL for the translations of the images')
    parser.add_argument('-s', '--schema_name', type=str,
                        default='ImageTranslationSchema',
                        help='how to name the data schema of the output'
                             'translations')
    parser.add_argument('translators', type=str, nargs='+',
                        help='one or more translators to apply to the images')

    log_group = parser.add_argument_group('Logging settings')
    cfg.LoggingConfig.setup_parser(log_group)

    cache_group = parser.add_argument_group('Cache settings')
    cfg.WebCacheConfig.setup_parser(cache_group)

    torch_group = parser.add_argument_group('Torch settings')
    cfg.TorchConfig.setup_parser(torch_group)

    peta_read_group = parser.add_argument_group('Settings for reading the '
                                                'input dataset')
    cfg.PetastormReadConfig.setup_parser(peta_read_group,
                                         default_batch_size=16)

    peta_write_group = parser.add_argument_group('Settings for writing the '
                                                 'output dataset')
    cfg.PetastormWriteConfig.setup_parser(peta_write_group,
                                          default_spark_master='local[*]',
                                          default_spark_driver_memory='40g',
                                          default_spark_exec_memory='20g',
                                          default_num_partitions=64 * 3,
                                          default_row_size='1024')

    return parser.parse_args()


def main(id_field: UnischemaField):
    """
    Translate images to low-dimensional, interpretable data.
    The images are read from a petastorm parquet store. In this parquet store,
    there must be two fields:

      - A field holding a unique id for each image.
        This field may have any name, it is passed with the parameter
        `id_field`.
      - A field holding image data encoded with the petastorm png encoder.
        This field must be named *image*.

    :param id_field: the field to use as unique image identifier
    """
    mp.set_start_method('spawn')

    args = _parse_args()

    cfg.LoggingConfig.set_from_args(args)
    torch_cfg = cfg.TorchConfig.from_args(args)
    read_cfg = cfg.PetastormReadConfig.from_args(args)
    write_cfg = cfg.PetastormWriteConfig.from_args(args)
    cache_dir = cfg.WebCacheConfig.from_args(args)

    spark_session = SparkSession.builder \
        .config('spark.driver.memory',
                write_cfg.spark_driver_memory) \
        .config('spark.executor.memory',
                write_cfg.spark_exec_memory) \
        .master(write_cfg.spark_master) \
        .getOrCreate()

    spark_session.sparkContext.setLogLevel('WARN')

    factory = TranslatorFactory(args.input_url, id_field, read_cfg, torch_cfg,
                                write_cfg, cache_dir)
    translators = [factory.create(t_spec) for t_spec in args.translators]
    data_gen = JoinDataGenerator(translators)
    out_df = data_gen()

    logging.info('Writing dataframe. First row: {}'.format(out_df.take(1)))

    output_url = args.output_url

    while True:
        try:
            with materialize_dataset(spark_session, output_url,
                                     data_gen.schema,
                                     write_cfg.row_group_size_mb):
                out_df.write.mode('error').parquet(args.output_url)
        except Exception as e:
            logging.error('Encountered exception: {}'.format(e))
            other_url = input('To retry, enter another '
                              'output URL and press <Enter>.'
                              'To exit, just press <Enter>.')
            if not other_url:
                raise e
            output_url = other_url
        else:
            break

    logging.info('----- Finished -----')
