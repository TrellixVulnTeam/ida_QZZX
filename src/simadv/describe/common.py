import abc
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from petastorm.unischema import Unischema, dict_to_spark_row
from pyspark.sql import DataFrame

from simadv.common import RowDict, LoggingMixin
from simadv.io import Field, Schema, PetastormReadConfig, PetastormWriteConfig


@dataclass
class ImageReadConfig(PetastormReadConfig):
    input_schema: Unischema = field(default=Schema.IMAGES, init=False)


@dataclass
class ImageDescriber(LoggingMixin, abc.ABC):

    read_cfg: ImageReadConfig
    write_cfg: PetastormWriteConfig
    concept_group_name: str

    def __post_init__(self):
        assert Field.IMAGE_ID in self.write_cfg.output_schema.fields

    @abc.abstractmethod
    def to_df(self) -> DataFrame:
        """
        Generates descriptions for all input images configured by `read_cfg`
        and returns them as a spark dataframe.
        """

    def to_parquet(self):
        """
        Generates descriptions for all input images configured by `read_cfg`
        and stores them in a Parquet store according to `write_cfg`.
        """
        self.write_cfg.write_parquet(self.to_df())


class DictBasedImageDescriber(ImageDescriber):

    # after which time to automatically stop
    time_limit_s: Optional[int] = None

    @abc.abstractmethod
    def generate(self) -> Iterator[RowDict]:
        """
        Generate row dicts that conform to `self.write_cfg.output_schema`.
        The row dicts should be yielded as soon as they are produced, for real-time logging.
        """

    def _generate_with_logging(self):
        with self._log_task('Describing with: {}'.format(self.concept_group_name)):
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
        rows = [dict_to_spark_row(self.write_cfg.output_schema, row_dict) for row_dict in self._generate_with_logging()]
        return self.write_cfg.session.createDataFrame(rows, self.write_cfg.output_schema.as_spark_schema())
