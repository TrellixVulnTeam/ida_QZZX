import abc
import argparse
import logging
import re
from collections import Counter
from contextlib import suppress
from typing import Callable

import numpy as np
import torch

import torchvision
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import UnischemaField, dict_to_spark_row, Unischema
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from torch.nn import DataParallel
from torchvision.transforms.functional import to_pil_image

from thesis_impl.places365.hub import Places365Hub
from thesis_impl.places365.io import unsupervised_loader
import thesis_impl.places365.config as cfg


class Translator(Callable, abc.ABC):

    @property
    @abc.abstractmethod
    def fields(self) -> [UnischemaField]:
        pass


class ImageToImageDummyTranslator(Translator):

    def __init__(self, width: int, height: int):
        self._field = UnischemaField('image', np.uint8, (width, height, 3),
                                     CompressedImageCodec('png'), False)
        self._width = width
        self._height = height

    @property
    def fields(self) -> [UnischemaField]:
        return [self._field]

    def __call__(self, image_tensors):
        for image_tensor in image_tensors.cpu():
            image_arr = np.asarray(to_pil_image(image_tensor, 'RGB')
                                   .resize((self._width, self._height)))
            yield {self._field.name: image_arr}


def _wrap_parallel(model):
    if torch.cuda.device_count() > 1:
        return DataParallel(model)
    return model


class ImageToPlaces365SceneNameTranslator(Translator):

    def __init__(self, model, hub: Places365Hub, top_k=1):
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
    def with_resnet18(device=cfg.Torch.DEFAULT_DEVICE, top_k=5, **kwargs):
        hub = Places365Hub(**kwargs)
        model = _wrap_parallel(hub.resnet18().to(device).eval())
        return ImageToPlaces365SceneNameTranslator(model, hub=hub, top_k=top_k)

    def __call__(self, image_tensors):
        with torch.no_grad():
            probs = self.model(image_tensors)
            _, predicted_labels_batch = torch.topk(probs, self._top_k, -1)
            for predicted_labels in predicted_labels_batch:
                yield {self._get_field_name(i): predicted_labels[i].item()
                       for i in range(self._top_k)}


class ImageToCocoObjectNamesTranslator(Translator):

    OBJECT_NAMES = [
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

    def __init__(self, model):
        self.model = model

        num_ints = int(len(self.OBJECT_NAMES) / 8) + 1
        self._field = UnischemaField('coco_object_counts', np.uint8,
                                     (num_ints,), NdarrayCodec(), False)

    @property
    def fields(self):
        return [self._field]

    @staticmethod
    def with_faster_r_cnn(device=cfg.Torch.DEFAULT_DEVICE):
        model = _wrap_parallel(torchvision.models.detection
                               .fasterrcnn_resnet50_fpn(pretrained=True)
                               .to(device)
                               .eval())
        return ImageToCocoObjectNamesTranslator(model)

    def get_object_names_counts(self, counts_array):
        """
        Returns a mapping from object names to counts,
        as encoded in *counts_array*.
        """
        return {self.OBJECT_NAMES[obj_id]: count
                for obj_id, count in counts_array.items() if count > 0}

    def __call__(self, image_tensors):
        with torch.no_grad():
            for result in self.model(image_tensors):
                obj_counts = Counter(obj_id.item()
                                     for obj_id in result['labels'])

                # uint8 must be smaller than 256
                assert max(*obj_counts.values()) < 256

                obj_ids = range(len(self.OBJECT_NAMES))
                counts_arr = np.array([obj_counts.get(obj_id, 0)
                                       for obj_id in obj_ids],
                                      dtype=np.uint8)
                yield {self._field.name: counts_arr}


_RE_IMAGE = re.compile(r'images\[(?P<width>\d+),\s?(?P<height>\d+)\]')
_RE_PLACES365 = re.compile(r'places365_scenes\[(?P<model>.*)\](\@(?P<k>\d))?')
_RE_COCO = re.compile(r'coco_objects\[(?P<model>.*)\]')


def translator_factory(t_name):
    image_match = _RE_IMAGE.fullmatch(t_name)
    if image_match:
        d = image_match.groupdict()
        width, height = int(d['width']), int(d['height'])
        return ImageToImageDummyTranslator(width, height)

    places365_match = _RE_PLACES365.fullmatch(t_name)
    if places365_match:
        d = places365_match.groupdict()
        model = d['model']
        params = {'top_k': int(d['k'])} if 'k' in d else {}

        if model in ['default', 'resnet18']:
            return ImageToPlaces365SceneNameTranslator.with_resnet18(**params)

        raise ValueError('Unknown model: {}'.format(model))

    coco_match = _RE_COCO.fullmatch(t_name)
    if coco_match:
        d = coco_match.groupdict()
        model = d['model']

        if model in ['default', 'faster_r_cnn']:
            return ImageToCocoObjectNamesTranslator.with_faster_r_cnn()

        raise ValueError('Unknown model: {}'.format(model))

    raise ValueError('Unknown translator: {}'.format(t_name))


def translate_images(input_url, output_url, schema_name,
                     translators: [Translator]):
    schema = Unischema(schema_name,
                       [f for t in translators for f in t.fields])

    # hub = Places365Hub()
    # with hub.open_demo_image() as demo_file:
    #     image = Image.open(demo_file).resize((224, 224))
    #     image_tensor = to_tensor(image).unsqueeze(0)
    #
    #     for d in data_gen(image_tensor):
    #         logging.info(d)

    try:
        loader = unsupervised_loader(input_url)

        spark = SparkSession.builder \
            .config('spark.driver.memory',
                    cfg.Petastorm.Write.spark_driver_memory) \
            .master(cfg.Petastorm.Write.spark_master) \
            .getOrCreate()

        sc = spark.sparkContext

        def translation_iterators(images_batch):
            for t in translators:
                yield t(images_batch)

        def generate_translations(images_batch):
            for result_row_dicts in zip(*translation_iterators(images_batch)):
                merged_row = {}
                for row_dict in result_row_dicts:
                    merged_row.update(row_dict)
                yield merged_row

        def create_rdd(images_batch):
            translations_batch = list(generate_translations(images_batch))
            return sc.parallelize(translations_batch) \
                .map(lambda x: dict_to_spark_row(schema, x))

        with suppress(StopIteration):
            images_batch = next(loader)
            translations_rdd = create_rdd(images_batch)

        for images_batch in loader:
            batch_rdd = create_rdd(images_batch)
            translations_rdd = translations_rdd.union(batch_rdd)
            logging.info('created new rdd. first item: {}'.
                         format(batch_rdd.take(1)))
            logging.info('current memory usage:\n{}'
                         .format(torch.cuda.memory_summary()))

        with materialize_dataset(spark, output_url, schema,
                                 cfg.Petastorm.Write.row_group_size_mb):
                spark.createDataFrame(translations_rdd,
                                      schema.as_spark_schema()) \
                    .coalesce(10) \
                    .write \
                    .mode('overwrite') \
                    .parquet(output_url)

    except KeyboardInterrupt:
        logging.info('---- ! Stopping due to KeyboardInterrupt ! ----')
    else:
        logging.info('----- Finished -----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate images to low'
                                                 'dimensional and interpretable'
                                                 'kinds of data.')
    parser.add_argument('-i', '--input_url', type=str,
                        help='input URL for the images')
    parser.add_argument('-o', '--output_url', type=str,
                        help='output URL for the translations of the images')
    parser.add_argument('-s','--schema_name', type=str,
                        default='ImageTranslationSchema',
                        help='how to name the data schema of the output'
                             'translations')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='whether to output more information for debugging')
    parser.add_argument('translators', type=str, nargs='+',
                        help='one or more translators to apply to the images')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    translators = [translator_factory(t_name) for t_name in args.translators]
    translate_images(args.input_url, args.output_url, args.schema_name,
                     translators)
