import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image
from petastorm.codecs import ScalarCodec, CompressedImageCodec

from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from thesis_impl import config as cfg
from thesis_impl.places365.hub import Places365Hub
from thesis_impl.util.webcache import WebCache


def _get_schema(image_width: int, image_height: int):
    return Unischema('Places365SupervisedSchema',
                     [UnischemaField('image', np.uint8, (image_width,
                                                         image_height, 3),
                                     CompressedImageCodec('png'), False),
                      UnischemaField('label_id', np.int16, (),
                                     ScalarCodec(IntegerType()), False)])


_RE_IMAGE_ID = re.compile(r'[0-9]{8}')
_RE_SIZE = re.compile(r'^[0-9]*x[0-9]*$')


class Converter:

    def __init__(self, images_dir: str, hub: Places365Hub,
                 write_cfg: cfg.PetastormWriteConfig):
        self.images_dir = Path(images_dir).expanduser()
        assert self.images_dir.exists()
        self.hub = hub
        self.write_cfg = write_cfg

    def _lookup_validation_label(self, image_path):
        image_file_name = image_path.name
        return self.hub.validation_label_map[image_file_name]

    def _lookup_train_label(self, image_path):
        two_parents = image_path.parents[2]
        image_file_name = '/' + str(image_path.relative_to(two_parents))
        return self.hub.train_label_map[image_file_name]

    def convert(self, image_size, glob='*.jpg', subset='validation',
                output_url=None):
        if subset == 'validation':
            lookup_label = self._lookup_validation_label
        elif subset == 'train':
            lookup_label = self._lookup_train_label
        else:
            raise NotImplementedError('Currently only images from the '
                                      'validation or train subset can be '
                                      'converted.')

        if output_url is None:
            output_url = 'file://' + str(self.images_dir.absolute() /
                                         '{}.parquet'.format(subset))

        spark = SparkSession.builder\
            .config('spark.driver.memory',
                    self.write_cfg.spark_driver_memory)\
            .master(self.write_cfg.spark_master)\
            .getOrCreate()

        sc = spark.sparkContext

        def generate_row(image_path: Path):
            label_id = lookup_label(image_path)
            image = Image.open(image_path).resize(image_size)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return {'image': np.asarray(image),
                    'label_id': label_id}

        schema = _get_schema(*image_size)

        with materialize_dataset(spark, output_url, schema,
                                 self.write_cfg.row_group_size_mb):
            rows_rdd = sc.parallelize(self.images_dir.glob(glob)) \
                .repartition(self.write_cfg.num_partitions) \
                .map(generate_row) \
                .map(lambda x: dict_to_spark_row(schema, x))

            spark.createDataFrame(rows_rdd, schema.as_spark_schema()) \
                .write \
                .mode('overwrite') \
                .parquet(output_url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a subset of the '
                                                 'Places365 data to a Parquet'
                                                 'store.')
    parser.add_argument('subset', type=str, choices=['validation', 'train'],
                        help='the subset of images to convert')
    parser.add_argument('images_dir', type=str,
                        help='the directory where the images are stored')
    parser.add_argument('images_glob', type=str,
                        help='glob expression specifying which images in the '
                             'above directory should be converted')
    parser.add_argument('--size', type=str, required=True,
                        help='Specify in which size the images are stored '
                             'as a string "[width]x[height]"')
    parser.add_argument('-o', '--output_url', type=str, default=None,
                        help='URL where to store the dataset')

    log_group = parser.add_argument_group('Logging settings')
    cfg.LoggingConfig.setup_parser(log_group)

    cache_group = parser.add_argument_group('Cache settings')
    cfg.WebCacheConfig.setup_parser(cache_group)

    peta_write_group = parser.add_argument_group('Settings for writing the '
                                                 'output dataset')
    cfg.PetastormWriteConfig.setup_parser(peta_write_group,
                                          default_spark_master='local[8]',
                                          default_spark_driver_memory='40g',
                                          default_num_partitions=64 * 3,
                                          default_row_size=1024)

    args = parser.parse_args()

    cfg.LoggingConfig.set_from_args(args)
    _cache_dir = cfg.WebCacheConfig.from_args(args)
    _write_cfg = cfg.PetastormWriteConfig.from_args(args)

    args = parser.parse_args()

    if _RE_SIZE.match(args.size) is None:
        raise ValueError('Format of --resize parameter must be '
                         '"[width]x[height]".')
    width, height = args.size.split('x')
    _size = int(width), int(height)

    _images_dir = args.images_dir.replace('\'', '')
    _images_glob = args.images_glob.replace('\'', '')

    _cache = WebCache(cfg.WebCacheConfig.from_args(args))
    _hub = Places365Hub(_cache)

    converter = Converter(_images_dir, _hub, _write_cfg)
    converter.convert(_size, _images_glob, args.subset, args.output_url)
