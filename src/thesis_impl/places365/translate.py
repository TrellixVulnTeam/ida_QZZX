import abc
import argparse
import csv
import logging
import re
from collections import Counter
from contextlib import suppress, contextmanager
from functools import reduce, partial
from typing import Callable

import numpy as np
import torch
import torchvision
from torch.nn import DataParallel
from torchvision.transforms.functional import to_pil_image


from petastorm import make_reader
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.tf_utils import make_petastorm_dataset
from petastorm.unischema import UnischemaField, dict_to_spark_row, Unischema

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType, FloatType

from thesis_impl.places365.hub import Places365Hub
from thesis_impl.places365.io import unsupervised_loader
import thesis_impl.places365.config as cfg
from thesis_impl.util.webcache import WebCache

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)  # avoid log spamming by tf
tf.autograph.set_verbosity(1)


class Translator(Callable, abc.ABC):

    def __init__(self, tensor_type: str, output_type_name: str):
        assert tensor_type in ['torch', 'tf']
        self.tensor_type = tensor_type
        self.output_type_name = output_type_name

    @property
    @abc.abstractmethod
    def fields(self) -> [UnischemaField]:
        pass


class ToImageDummyTranslator(Translator):

    def __init__(self, width: int, height: int):
        super().__init__('torch', 'the untouched original image')
        self._field = UnischemaField('image', np.uint8, (width, height, 3),
                                     CompressedImageCodec('png'), False)
        self._width = width
        self._height = height

    @property
    def fields(self) -> [UnischemaField]:
        return [self._field]

    def __call__(self, image_tensors):
        with torch.no_grad():
            for image_tensor in image_tensors.cpu():
                image_arr = np.asarray(to_pil_image(image_tensor, 'RGB')
                                       .resize((self._width, self._height)))
                yield {self._field.name: image_arr}


class ToColorDistributionTranslator(Translator):

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

    def __init__(self, num_colors: int=10, resize_to: int=100):
        super().__init__('torch', 'distribution of colors in the image')
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

    def __call__(self, image_tensors):
        with torch.no_grad():
            for image_tensor in image_tensors.cpu():
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

                yield {self._get_field_name(color): fraction
                       for color, fraction in color_fractions.items()}


def _wrap_parallel(model):
    if torch.cuda.device_count() > 1:
        return DataParallel(model)
    return model


class ToPlaces365SceneNameTranslator(Translator):

    def __init__(self, model, hub: Places365Hub, top_k=1):
        super().__init__('torch', 'scene names from the Places365 set')
        self.model = model
        self.hub = hub
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
    def with_resnet18(device: torch.device, cache=None, top_k=5):
        hub = Places365Hub(cache)
        model = _wrap_parallel(hub.resnet18().to(device).eval())
        return ToPlaces365SceneNameTranslator(model, hub=hub, top_k=top_k)

    def __call__(self, image_tensors):
        image_tensors = image_tensors.clone().detach()

        with torch.no_grad():
            probs = self.model(image_tensors)
            _, predicted_labels_batch = torch.topk(probs, self._top_k, -1)
            for predicted_labels in predicted_labels_batch:
                yield {f.name: predicted_labels[i].cpu().item()
                       for i, f in enumerate(self._fields)}


class ToCocoObjectNamesTranslator(Translator):

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

    def __init__(self, model, threshold: float):
        super().__init__('torch', 'detected objects from the COCO set')
        self.model = model

        assert 0 <= threshold < 1
        self.threshold = threshold

        num_ints = len(self.object_names())
        self._field = UnischemaField('coco_object_counts', np.uint8,
                                     (num_ints,), NdarrayCodec(), False)

    @property
    def fields(self):
        return [self._field]

    @staticmethod
    def object_names():
        return ToCocoObjectNamesTranslator._OBJECT_NAMES

    @staticmethod
    def with_faster_r_cnn(device: torch.device, threshold=.5):
        model = torchvision.models.detection\
            .fasterrcnn_resnet50_fpn(pretrained=True)\
            .to(device)\
            .eval()
        return ToCocoObjectNamesTranslator(model, threshold=threshold)

    @staticmethod
    def get_object_names_counts(counts_array):
        """
        Returns a mapping from object names to counts,
        as encoded in *counts_array*.
        """
        return {ToCocoObjectNamesTranslator.object_names()[obj_id]: count
                for obj_id, count in enumerate(counts_array) if count > 0}

    def __call__(self, image_tensors):
        image_tensors = image_tensors.clone().detach()

        with torch.no_grad():
            for result in self.model(image_tensors):
                obj_ids, scores = result['labels'], result['scores']
                obj_counts = Counter(obj_id.cpu().item()
                                     for i, obj_id in enumerate(obj_ids)
                                     if scores[i] > self.threshold)

                obj_ids = range(len(self.object_names()))
                counts_arr = np.array([obj_counts.get(obj_id, 0)
                                       for obj_id in obj_ids],
                                      dtype=np.uint8)
                yield {self._field.name: counts_arr}


class ToOIV4ObjectNameTranslator(Translator):

    _MODELS_URL = 'http://download.tensorflow.org/models/object_detection/'

    _LABELS_URL = 'https://storage.googleapis.com/openimages/2018_04/'

    def __init__(self, model_name: str, threshold: float, cache: WebCache):
        super().__init__('tf', 'detected objects from the OpenImages set')

        assert 0 <= threshold < 1
        self.threshold = threshold

        self.cache = cache

        num_ints = len(self.object_names())
        self._field = UnischemaField('oi_object_counts', np.uint8,
                                     (num_ints,), NdarrayCodec(), False)

        self._load_model(model_name)

    def _load_model(self, model_name: str):
        self.model_name = model_name
        model_file_name = model_name + '.tar.gz'

        self.cache.cache(model_file_name, self._MODELS_URL, is_archive=True)
        model_dir = self.cache.get_absolute_path(model_name) \
            / "saved_model"

        self.dist_strategy = tf.distribute.MirroredStrategy()
        with self.dist_strategy.scope():
            model = tf.saved_model.load(str(model_dir))
            self.model = model.signatures['serving_default']

    def object_names(self):
        return self.object_names_from_cache(self.cache)

    @staticmethod
    def object_names_from_cache(cache: WebCache):
        url = ToOIV4ObjectNameTranslator._LABELS_URL

        with cache.open('class-descriptions-boxable.csv', url)\
                as object_names_file:
            csv_reader = csv.reader(object_names_file, delimiter=',')
            return ['__background__'] + [row[1] for row in csv_reader]

    @property
    def fields(self) -> [UnischemaField]:
        return [self._field]

    def __call__(self, image_tensors):
        results = self.model(image_tensors)
        obj_ids_batches, scores_batches = results['detection_classes'].numpy(),\
                                          results['detection_scores'].numpy()
        for obj_ids, scores in zip(obj_ids_batches, scores_batches):
            obj_counts = Counter(obj_id for i, obj_id in enumerate(obj_ids)
                                 if scores[i] > self.threshold)

            obj_ids = range(len(self.object_names()))
            counts_arr = np.array([obj_counts.get(obj_id, 0)
                                   for obj_id in obj_ids],
                                  dtype=np.uint8)
            yield {self._field.name: counts_arr}


_RE_IMAGE = re.compile(r'(?P<name>images)\[(?P<width>\d+),\s?(?P<height>\d+)\]')
_RE_COLORS = re.compile(r'(?P<name>colors)(@(?P<k>\d+))?')
_RE_PLACES365 = re.compile(r'(?P<name>places365_scenes)\[(?P<model>.*)\]'
                           r'(@(?P<k>\d+))?')
_RE_COCO = re.compile(r'(?P<name>coco_objects)\[(?P<model>.*)\]'
                      r'(@(?P<t>0?\.\d+))?')
_RE_OI = re.compile(r'(?P<name>oi_objects)\[(?P<model>.*)\]'
                    r'(@(?P<t>0?\.\d+))?')


class TranslationProcessor:
    def __init__(self, input_url, output_url, schema_name,
                 translator_specs: [str], cache: WebCache,
                 torch_device: torch.device, read_cfg: cfg.PetastormReadConfig,
                 write_cfg: cfg.PetastormWriteConfig):
        self.input_url, self.output_url = input_url, output_url
        self.cache = cache
        self.torch_device = torch_device
        self.read_cfg, self.write_cfg = read_cfg, write_cfg

        self.translators_by_type = {'torch': [], 'tf': []}
        translator_fields = []

        for t_spec in translator_specs:
            t = self._create_translator(t_spec)
            self.translators_by_type[t.tensor_type].append(t)
            for f in t.fields:
                translator_fields.append(f)

        self._id_field = UnischemaField('translation_id', np.int64, (),
                                        ScalarCodec(IntegerType()), False)
        self.schema = Unischema(schema_name,
                                [self._id_field] + translator_fields)

        self._log_nesting = 0

    def _log(self, msg):
        prefix = '--' * self._log_nesting + ' ' if self._log_nesting else ''
        logging.info(prefix + msg)

    def _log_item(self, msg):
        self._log('<{}/>'.format(msg))

    def _log_group_start(self, msg):
        self._log_item(msg)
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

    def _create_translator(self, t_spec):
        def _raise_unknown_model(_model, _translator):
            raise ValueError('Unknown model {} for translator {}'
                             .format(_model, _translator))

        image_match = _RE_IMAGE.fullmatch(t_spec)
        if image_match:
            d = image_match.groupdict()
            width, height = int(d['width']), int(d['height'])
            return ToImageDummyTranslator(width, height)

        color_match = _RE_COLORS.fullmatch(t_spec)
        if color_match:
            d = color_match.groupdict()
            params = {'num_colors': int(d['k'])} if d['k'] else {}
            return ToColorDistributionTranslator(**params)

        places365_match = _RE_PLACES365.fullmatch(t_spec)
        if places365_match:
            d = places365_match.groupdict()
            model = d['model']
            params = {'top_k': int(d['k'])} if d['k'] else {}

            if model in ['default', 'resnet18']:
                return ToPlaces365SceneNameTranslator \
                    .with_resnet18(self.torch_device, **params)

            _raise_unknown_model(model, d['name'])

        coco_match = _RE_COCO.fullmatch(t_spec)
        if coco_match:
            d = coco_match.groupdict()
            model = d['model']
            params = {'threshold': float(d['t'])} if d['t'] else {}

            if model in ['default', 'faster_r_cnn']:
                return ToCocoObjectNamesTranslator \
                    .with_faster_r_cnn(self.torch_device, **params)

            _raise_unknown_model(model, d['name'])

        oi_match = _RE_OI.fullmatch(t_spec)
        if oi_match:
            d = oi_match.groupdict()
            model_short = d['model']
            params = {'threshold': float(d['t'])} if d['t'] else {}

            model_full = {'inception_resnet': 'faster_rcnn_inception_resnet_v2'
                                              '_atrous_oid_v4_2018_12_12',
                          'mobilenet': 'ssd_mobilenet_v2_oid_v4_2018_12_12'}
            model_full['default'] = model_full['inception_resnet']

            try:
                model = model_full[model_short]
            except KeyError:
                _raise_unknown_model(model_short, d['name'])
            else:
                return ToOIV4ObjectNameTranslator(model, cache=self.cache,
                                                  **params)

        raise ValueError('Unknown translator: {}'.format(t_spec))

    def _translate_to_df(self, spark_session, spark_ctx, tensor_type,
                         translators):
        with self._log_task('Generating translations based on {} tensors'
                            .format(tensor_type)):
            batches_iter = {'torch': self._get_torch_loader,
                            'tf': self._get_tf_loader}[tensor_type]()

            translator_fields = [f for t in translators for f in t.fields]
            partial_schema = Unischema(tensor_type, [self._id_field] +
                                       translator_fields)
            partial_spark_schema = partial_schema.as_spark_schema()

            def translation_iterators(images_batch):
                def log_then_call(t):
                    self._log_item('Translating to: {}'
                                   .format(t.output_type_name))
                    yield from t(images_batch)

                for _t in translators:
                    yield log_then_call(_t)

            def generate_translations(images_batch):
                stream = zip(*translation_iterators(images_batch))

                for t_id, result_row_dicts in enumerate(stream):
                    merged_row = {self._id_field.name: t_id}
                    for row_dict in result_row_dicts:
                        merged_row.update(row_dict)
                    yield dict_to_spark_row(partial_schema, merged_row)

            def create_df(images_batch):
                with self._log_task('Generating new batch of translations'):
                    t_batch = list(generate_translations(images_batch))
                    return spark_session.createDataFrame(t_batch,
                                                         partial_spark_schema)

            with suppress(StopIteration):
                _images_batch = next(batches_iter)
                translations = [create_df(_images_batch)]
                del _images_batch  # free memory of tensors

            for _images_batch in batches_iter:
                translations.append(create_df(_images_batch))
                del _images_batch

            self._log_item('Unionizing dataframes')
            return reduce(DataFrame.union, translations)

    def _get_torch_loader(self):
        return unsupervised_loader(self.input_url, self.read_cfg,
                                   self.torch_device)

    def _get_tf_loader(self):
        shuffle = self.read_cfg.shuffle_row_groups
        reader = make_reader(self.input_url, schema_fields=['image'],
                             shuffle_row_groups=shuffle)
        peta_dataset = make_petastorm_dataset(reader)\
            .batch(self.read_cfg.batch_size)

        def tensor_iter():
            for schema_view in iter(peta_dataset):
                yield schema_view.image

        return tensor_iter()

    def translate(self):
        spark_session = SparkSession.builder \
            .config('spark.driver.memory',
                    self.write_cfg.spark_driver_memory) \
            .config('spark.executor.memory',
                    self.write_cfg.spark_exec_memory) \
            .master(self.write_cfg.spark_master) \
            .getOrCreate()

        spark_ctx = spark_session.sparkContext
        spark_ctx.setLogLevel('WARN')

        dfs = [self._translate_to_df(spark_session, spark_ctx,
                                     t_type, t_list)
               for t_type, t_list in self.translators_by_type.items()]

        with materialize_dataset(spark_session, self.output_url, self.schema,
                                 self.write_cfg.row_group_size_mb):
            with self._log_task('Joining the dataframes of all '
                                'translators'):
                final_df = reduce(partial(DataFrame.join,
                                          on=self._id_field.name,
                                          how='inner'), dfs)
            with self._log_task('Writing to parquet store'):
                while True:
                    try:
                        final_df.write.mode('error')\
                            .parquet(self.output_url)
                    except Exception as e:
                        self._log('Encountered exception: {}'.format(e))
                        other_url = input('To retry, enter another '
                                          'output URL and press <Enter>.'
                                          'To exit, just press <Enter>.')
                        self.output_url = other_url
                        if not other_url:
                            raise e
                    else:
                        break

        logging.info('----- Finished -----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate images to low '
                                                 'dimensional and '
                                                 'interpretable kinds of data.')
    parser.add_argument('-i', '--input_url', type=str, required=True,
                        help='input URL for the images')
    parser.add_argument('-o', '--output_url', type=str, required=True,
                        help='output URL for the translations of the images')
    parser.add_argument('-s','--schema_name', type=str,
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

    args = parser.parse_args()

    cfg.LoggingConfig.set_from_args(args)
    _torch_cfg = cfg.TorchConfig.from_args(args)
    _read_cfg = cfg.PetastormReadConfig.from_args(args)
    _write_cfg = cfg.PetastormWriteConfig.from_args(args)

    _cache = WebCache(cfg.WebCacheConfig.from_args(args))
    processor = TranslationProcessor(args.input_url, args.output_url,
                                     args.schema_name, args.translators,
                                     _cache, _torch_cfg.device, _read_cfg,
                                     _write_cfg)
    processor.translate()
