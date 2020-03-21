import argparse
import logging
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from petastorm.codecs import ScalarCodec, CompressedImageCodec

from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from thesis_impl.places365 import wideresnet
from thesis_impl.places365.hub import Places365Hub


_Places365SupervisedSchema = Unischema('Places365SupervisedSchema', [
   UnischemaField('image_id', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('image', np.uint8, (256, 256, 3),
                  CompressedImageCodec('jpeg'), False),
   UnischemaField('label_id', np.uint8, (), ScalarCodec(IntegerType()), False),
])


_RE_IMAGE_ID = re.compile(r'[0-9]{8}')


class Converter:

    def __init__(self, images_dir: str, hub: Places365Hub):
        self.images_dir = Path(images_dir)
        assert self.images_dir.exists()
        self.hub = hub

    def convert(self, glob='*.jpg', subset='validation', output_url=None,
                transform=None, row_group_size_mb=1024,
                spark_driver_memory='8g', spark_master='local[8]'):
        if subset == 'validation':
            label_map = self.hub.validation_label_map
        elif subset == 'train':
            label_map = self.hub.train_label_map
        else:
            raise NotImplementedError('Currently only images from the '
                                      'validation or train subset can be '
                                      'converted.')

        if output_url is None:
            output_url = 'file://' + str(self.images_dir.absolute() /
                                         '{}.parquet'.format(subset))

        spark = SparkSession.builder\
            .config('spark.driver.memory', spark_driver_memory)\
            .master(spark_master)\
            .getOrCreate()

        sc = spark.sparkContext

        def generate_row(image_path: Path):
            image_file_name = image_path.name
            image_id = int(_RE_IMAGE_ID.search(image_file_name).group())
            label_id = label_map[image_file_name]

            image = Image.open(image_path)
            if transform == 'resnet18':
                image = wideresnet.IMAGE_TRANSFORM(image)

            return {'image_id': image_id,
                    'image': np.asarray(image),
                    'label_id': label_id}

        with materialize_dataset(spark, output_url, _Places365SupervisedSchema,
                                 row_group_size_mb):

            rows_rdd = sc.parallelize(self.images_dir.glob(glob)) \
                .map(generate_row) \
                .map(lambda x: dict_to_spark_row(_Places365SupervisedSchema, x))

            spark_schema = _Places365SupervisedSchema.as_spark_schema()
            spark.createDataFrame(rows_rdd, spark_schema) \
                .coalesce(10) \
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
    parser.add_argument('--transform', type=str, default=None,
                        help='Set to "resnet18" to transform the images as '
                             'required by the pretrained resnet18.')
    parser.add_argument('--output_url', type=str, default=None,
                        help='URL where to store the dataset')
    parser.add_argument('--debug', action='store_true',
                        help='whether to output more information for debugging')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='the directory where Places356 metadata is cached')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    hub = Places365Hub(args.cache_dir) if args.cache_dir else Places365Hub()
    converter = Converter(args.images_dir, hub)
    converter.convert(args.images_glob, args.subset, args.output_url,
                      args.transform)
