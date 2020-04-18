import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
from petastorm.codecs import CompressedImageCodec

from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import SparkSession

from thesis_impl import config as cfg
from thesis_impl.hub import SupervisedImageDataset
from thesis_impl.util.webcache import WebCache


class Converter:

    def __init__(self, images_dir: str, hub: SupervisedImageDataset,
                 write_cfg: cfg.PetastormWriteConfig):
        self.images_dir = Path(images_dir).expanduser()
        assert self.images_dir.exists()
        self.hub = hub
        self.write_cfg = write_cfg

    def _get_schema(self, image_width: int, image_height: int):
        image_data_field = UnischemaField('image', np.uint8, (None, None, 3),
                                          CompressedImageCodec('png'), False)

        return Unischema('{}SupervisedSchema'.format(self.hub.dataset_name),
                         [self.hub.image_id_field,
                          image_data_field,
                          self.hub.label_field])

    def convert(self, image_size, glob='*.jpg', subset='validation',
                output_url=None):
        assert subset in ['validation', 'train']

        if output_url is None:
            output_url = 'file://' + str(self.images_dir.absolute() /
                                         '{}.parquet'.format(subset))

        spark_session = SparkSession.builder\
            .config('spark.driver.memory',
                    self.write_cfg.spark_driver_memory)\
            .master(self.write_cfg.spark_master)\
            .getOrCreate()

        spark_ctx = spark_session.sparkContext

        def generate_row(image_path: Path):
            image_id = self.hub.get_image_id(image_path, subset)
            label = self.hub.get_image_label(image_path, subset)
            image = Image.open(image_path)
            image.thumbnail(image_size)  # maintains aspect ratio
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return {'image': np.asarray(image),
                    self.hub.image_id_field.name: image_id,
                    self.hub.label_field.name: label}

        schema = self._get_schema(*image_size)

        with materialize_dataset(spark_session, output_url, schema,
                                 self.write_cfg.row_group_size_mb):
            rows_rdd = spark_ctx.parallelize(self.images_dir.glob(glob)) \
                .repartition(self.write_cfg.num_partitions) \
                .map(generate_row) \
                .map(lambda x: dict_to_spark_row(schema, x))

            spark_session.createDataFrame(rows_rdd, schema.as_spark_schema()) \
                .write \
                .mode('overwrite') \
                .parquet(output_url)


def main(parser: ArgumentParser, hub_class):
    conv_group = parser.add_argument_group('Conversion settings')
    cfg.ConverterConfig.setup_parser(conv_group)

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

    conv_cfg = cfg.ConverterConfig.from_args(args)
    cfg.LoggingConfig.set_from_args(args)
    write_cfg = cfg.PetastormWriteConfig.from_args(args)

    cache = WebCache(cfg.WebCacheConfig.from_args(args))
    hub = hub_class(cache)

    converter = Converter(conv_cfg.images_dir, hub, write_cfg)
    converter.convert(conv_cfg.size, conv_cfg.images_glob, conv_cfg.subset,
                      conv_cfg.output_url)
