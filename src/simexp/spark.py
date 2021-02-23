import abc
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from functools import reduce
from typing import Optional, List, Iterator, Any

import numpy as np
import pyspark.sql.functions as sf
import pyspark.sql.types as st
from petastorm import make_reader
from petastorm.codecs import CompressedImageCodec, ScalarCodec, CompressedNdarrayCodec
from petastorm.etl.dataset_metadata import get_schema_from_dataset_url, materialize_dataset
from petastorm.tf_utils import make_petastorm_dataset
from petastorm.unischema import UnischemaField, Unischema, dict_to_spark_row
from pyspark.sql import SparkSession, DataFrame

from simexp.common import RowDict, LoggingMixin, ComposableDataclass


class Field(UnischemaField, Enum):
    """
    All data fields used by the different submodules.
    """
    IMAGE = UnischemaField('image', np.uint8, (None, None, 3), CompressedImageCodec('png'), False)
    IMAGE_ID = UnischemaField('image_id', np.unicode_, (), ScalarCodec(st.StringType()), False)
    DESCRIBER = UnischemaField('describer', np.unicode_, (), ScalarCodec(st.StringType()), False)
    CONCEPT_NAMES = UnischemaField('concept_names',  np.unicode_, (None,), CompressedNdarrayCodec(), False)
    CONCEPT_MASKS = UnischemaField('concept_masks', np.bool_, (None, None, None), CompressedNdarrayCodec(), False)
    INFLUENCE_MASK = UnischemaField('influence_mask', np.float_, (None, None), CompressedNdarrayCodec(), False)
    PREDICTED_CLASS = UnischemaField('predicted_class', np.uint16, (), ScalarCodec(st.IntegerType()), False)
    INFLUENCE_ESTIMATOR = UnischemaField('influence_estimator', np.unicode_, (), ScalarCodec(st.StringType()), True)
    PERTURBER = UnischemaField('perturber', np.unicode_, (), ScalarCodec(st.StringType()), True)
    DETECTOR = UnischemaField('detector', np.unicode_, (), ScalarCodec(st.StringType()), True)
    DESCRIBERS = UnischemaField('describers', np.unicode_, (None,), CompressedNdarrayCodec(), False)
    CONCEPT_COUNTS = UnischemaField('concept_counts', np.uint8, (None,), CompressedNdarrayCodec(), False)
    PERTURBED_IMAGE_ID = UnischemaField('perturbed_image_id', np.unicode_, (), ScalarCodec(st.StringType()), True)

    def encode(self, value: Any) -> Any:
        return self.codec.encode(self, value)

    def decode(self, encoded: Any) -> Any:
        """
        Factored out from `petastorm.utils.decode_row()`.
        """
        if self.codec:
            return self.codec.decode(self, encoded)
        elif self.numpy_dtype and issubclass(self.numpy_dtype, (np.generic, Decimal)):
            return self.numpy_dtype(encoded)
        else:
            return encoded

    def decode_from_row_dict(self, row_dict: RowDict) -> Any:
        return self.decode(row_dict[self.name])


class Schema:
    """
    All data schemas used by the different submodules.
    """
    IMAGES = Unischema('Images', [Field.IMAGE_ID, Field.IMAGE])
    CONCEPT_MASKS = Unischema('ConceptMasks', [Field.IMAGE_ID, Field.DESCRIBER, Field.CONCEPT_NAMES,
                                               Field.CONCEPT_MASKS])
    PIXEL_INFLUENCES = Unischema('PixelInfluences', [Field.IMAGE_ID, Field.PREDICTED_CLASS,
                                                     Field.INFLUENCE_MASK, Field.INFLUENCE_ESTIMATOR])


@dataclass
class PetastormReadConfig:
    input_schema: Unischema
    input_url: str
    shuffle: bool = False
    pool_type: str = 'process'
    workers_count: int = 5

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
        return make_petastorm_dataset(self.make_reader(schema_fields))


@dataclass
class PetastormWriteConfig:
    output_schema: Unischema
    output_url: str
    row_size: int


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

    def write_petastorm(self, out_df: DataFrame, write_cfg: PetastormWriteConfig):
        logging.info('Writing {} rows to petastorm parquet store.'.format(out_df.count()))

        output_url = write_cfg.output_url
        while True:
            try:
                with materialize_dataset(self.session, output_url,
                                         write_cfg.output_schema, write_cfg.row_size):
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


@dataclass
class DataGenerator(LoggingMixin, abc.ABC):

    @abc.abstractmethod
    def to_df(self) -> DataFrame:
        """
        Generates descriptions for all input images configured by `read_cfg`
        and returns them as a spark dataframe.
        """


@dataclass
class DictBasedDataGenerator(DataGenerator):

    # unique identifier for this describer
    name: str

    # how to interpret generated row dicts
    output_schema: Unischema

    # spark session for creating the result dataframe
    spark_cfg: SparkSessionConfig

    # after which time to automatically stop
    time_limit_s: Optional[int]

    @abc.abstractmethod
    def generate(self) -> Iterator[RowDict]:
        """
        Generate row dicts that conform to `self.write_cfg.output_schema`.
        The row dicts should be yielded as soon as they are produced, for real-time logging.
        """

    def _generate_with_logging(self):
        with self._log_task('Describing with: {}'.format(self.name)):
            start_time = time.time()
            last_time = start_time

            for num_rows, row_dict in enumerate(self.generate()):
                current_time = time.time()
                if current_time - last_time > 5:
                    self._log_item('Described {} rows so far.'
                                   .format(num_rows + 1))
                    last_time = current_time
                yield row_dict

                if self.time_limit_s is not None and current_time - start_time > self.time_limit_s:
                    self._log_item('Reached timeout! Stopping.')
                    break

    def to_df(self) -> DataFrame:
        rows = [dict_to_spark_row(self.output_schema, row_dict) for row_dict in self._generate_with_logging()]
        return self.spark_cfg.session.createDataFrame(rows, self.output_schema.as_spark_schema())


@dataclass
class ConceptMasksUnion(ComposableDataclass):

    # how to use spark
    spark_cfg: SparkSessionConfig

    # urls of petastorm parquet stores of schema `Schema.CONCEPT_MASKS`
    concept_mask_urls: List[str]

    def __post_init__(self):
        super().__post_init__()
        self.union_df = self._get_union_of_describers_df()

        explode_names = sf.explode(sf.flatten(sf.col(Field.CONCEPT_NAMES.name))).alias(Field.CONCEPT_NAMES.name)
        all_concept_names_df = self.union_df.select(explode_names).distinct()
        self.all_concept_names = [row[Field.CONCEPT_NAMES.name] for row in all_concept_names_df.collect()]

    def _get_union_of_describers_df(self):
        @sf.udf(st.ArrayType(st.StringType()))
        def unique_cleaned_concept_names(describer_name, concept_names):
            concept_names = Field.CONCEPT_NAMES.decode(concept_names)
            concept_names = (describer_name + '.' + np.char.asarray(np.unique(concept_names))).lower().tolist()

            cleaned_concept_names = []
            for concept_name in concept_names:
                for c in ' ,;{}()\n\t=':
                    if c in concept_name:
                        concept_name = concept_name.replace(c, '_')

                cleaned_concept_names.append(concept_name)

            return cleaned_concept_names

        return reduce(DataFrame.union, [self.spark_cfg.session.read.parquet(url) for url in self.concept_mask_urls]) \
            .withColumn('tmp', unique_cleaned_concept_names(Field.DESCRIBER.name,
                                                            Field.CONCEPT_NAMES.name)) \
            .drop(Field.DESCRIBER.name, Field.CONCEPT_NAMES.name) \
            .withColumnRenamed('tmp', Field.CONCEPT_NAMES.name) \
            .groupBy(Field.IMAGE_ID.name) \
            .agg(*[sf.collect_list(sf.col(f.name)).alias(f.name)
                   for f in [Field.CONCEPT_NAMES, Field.CONCEPT_MASKS]])
