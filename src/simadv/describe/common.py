import abc
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from petastorm.unischema import Unischema, dict_to_spark_row
from pyspark.sql import DataFrame

from simadv.common import RowDict, LoggingMixin
from simadv.io import Schema, PetastormReadConfig, SparkSessionConfig


@dataclass
class ImageReadConfig(PetastormReadConfig):
    input_schema: Unischema = field(default=Schema.IMAGES, init=False)


@dataclass
class ImageDescriber(LoggingMixin, abc.ABC):

    # unique identifier for this describer
    name: str

    @abc.abstractmethod
    def to_df(self) -> DataFrame:
        """
        Generates descriptions for all input images configured by `read_cfg`
        and returns them as a spark dataframe.
        """


@dataclass
class DictBasedImageDescriber(ImageDescriber):

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
