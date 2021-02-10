import abc
import logging
from dataclasses import dataclass
from typing import Optional, List, Iterable

import numpy as np
from petastorm import make_reader
from petastorm.codecs import CompressedImageCodec, ScalarCodec, NdarrayCodec, CompressedNdarrayCodec
from petastorm.etl.dataset_metadata import get_schema_from_dataset_url, materialize_dataset
from petastorm.tf_utils import make_petastorm_dataset
from petastorm.unischema import UnischemaField, Unischema
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, IntegerType


class Field:
    """
    All data fields used by the different submodules.
    """
    IMAGE = UnischemaField('image', np.uint8, (None, None, 3), CompressedImageCodec('png'), False)
    IMAGE_ID = UnischemaField('image_id', np.unicode_, (),
                              ScalarCodec(StringType()), False)
    CONCEPT_GROUP = UnischemaField('concept_group', np.unicode_, (), ScalarCodec(StringType()), False)
    CONCEPT_NAMES = UnischemaField('concept_names',  np.unicode_, (None,), ScalarCodec(StringType()), False)
    CONCEPT_MASKS = UnischemaField('concept_masks', np.bool, (None, None, None), NdarrayCodec(), False)
    INFLUENCE_MASK = UnischemaField('influence_mask', np.float32, (None, None), CompressedNdarrayCodec(), False)
    PREDICTED_CLASS = UnischemaField('predicted_class', np.uint16, (), ScalarCodec(IntegerType()), False)
    INFLUENCE_ESTIMATOR = UnischemaField('influence_estimator', np.unicode_, (), ScalarCodec(StringType()), True)
    PERTURBER = UnischemaField('perturber', np.unicode_, (), ScalarCodec(StringType()), True)
    DETECTOR = UnischemaField('detector', np.unicode_, (), ScalarCodec(StringType()), True)
    OBJECT_COUNTS = UnischemaField('object_counts', np.uint8, (None,), CompressedNdarrayCodec(), False)
    PERTURBED_IMAGE_ID = UnischemaField('perturbed_image_id', np.unicode_, (), ScalarCodec(StringType()), False)


class Schema:
    """
    All data schemas used by the different submodules.
    """
    IMAGES = Unischema('Images', [Field.IMAGE_ID, Field.IMAGE])
    TEST = Unischema('Test', [Field.IMAGE_ID, Field.OBJECT_COUNTS, Field.PREDICTED_CLASS])
    CONCEPT_MASKS = Unischema('ConceptMasks', [Field.IMAGE_ID, Field.CONCEPT_GROUP, Field.CONCEPT_NAMES,
                                               Field.CONCEPT_MASKS])
    PIXEL_INFLUENCES = Unischema('PixelInfluences', [Field.IMAGE_ID, Field.PREDICTED_CLASS,
                                                     Field.INFLUENCE_MASK, Field.INFLUENCE_ESTIMATOR])
    PERTURBED_OBJECT_COUNTS = Unischema('PerturbedObjectCounts',
                                        [Field.IMAGE_ID, Field.OBJECT_COUNTS, Field.PREDICTED_CLASS,
                                         Field.INFLUENCE_ESTIMATOR, Field.PERTURBER, Field.DETECTOR,
                                         Field.PERTURBED_IMAGE_ID])


@dataclass
class PetastormReadConfig:
    input_schema: Unischema
    input_url: str
    batch_size: int
    shuffle: bool = False
    pool_type: str = 'thread'
    workers_count: int = 10

    def make_reader(self, schema_fields: Optional[List[str]] = None, **kwargs):
        actual_schema = get_schema_from_dataset_url(self.input_url)
        assert set(actual_schema.fields) >= set(self.input_schema.fields)

        return make_reader(self.input_url,
                           shuffle_row_groups=self.shuffle,
                           reader_pool_type=self.pool_type,
                           workers_count=self.workers_count,
                           schema_fields=schema_fields,
                           **kwargs)

    def make_tf_dataset(self, schema_fields: Optional[List[str]] = None):
        return make_petastorm_dataset(self.make_reader(schema_fields)).batch(self.batch_size)


@dataclass
class SparkSessionConfig:
    spark_master: str
    spark_driver_memory: str
    spark_exec_memory: str

    def __post_init__(self):
        self.builder = SparkSession.builder \
            .config('spark.driver.memory',
                    self.spark_driver_memory) \
            .config('spark.executor.memory',
                    self.spark_exec_memory) \
            .master(self.spark_master)

    @property
    def session(self):
        return self.builder.getOrCreate()


@dataclass
class PetastormWriteConfig(SparkSessionConfig):
    output_schema: Unischema
    output_url: str
    row_size: int

    def write_parquet(self, out_df: DataFrame):
        logging.info('Writing {} object count observations to petastorm parquet store.'.format(out_df.count()))

        output_url = self.output_url
        while True:
            try:
                with materialize_dataset(self.session, output_url,
                                         self.output_schema, self.row_size):
                    out_df.write.mode('error').parquet(output_url)
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

