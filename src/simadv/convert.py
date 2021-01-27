import logging
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Union, Optional, Tuple, Type

import numpy as np
from PIL import Image
from petastorm.codecs import CompressedImageCodec

from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from simple_parsing import ArgumentParser

from simadv.common import PetastormWriteConfig, LoggingConfig
from simadv.hub import SupervisedImageDataset


@dataclass
class ConvertTask:

    images_dir: Union[str, Path]
    write_cfg: PetastormWriteConfig
    hub: SupervisedImageDataset

    glob: str = '*.jpg'
    sample_size: Optional[int] = None
    subset: str = 'validation'
    output_url: Optional[str] = None
    ignore_missing_labels: bool = True
    image_size: Optional[Tuple[int, int]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    num_partitions: int = 64 * 3

    def __post_init__(self):
        if isinstance(self.images_dir, str):
            self.images_dir = Path(self.images_dir)
        self.images_dir = self.images_dir.expanduser()
        assert self.images_dir.exists()

        assert self.subset in ['validation', 'test']
        if self.output_url is None:
            self.output_url = 'file://' + str(self.images_dir.absolute() /
                                              '{}.parquet'.format(self.subset))

    def _get_schema(self):
        image_data_field = UnischemaField('image', np.uint8, (None, None, 3),
                                          CompressedImageCodec('png'), False)

        return Unischema('{}SupervisedSchema'.format(self.hub.dataset_name),
                         [self.hub.image_id_field,
                          image_data_field,
                          self.hub.label_field])

    def run(self):
        spark_ctx = self.write_cfg.session.sparkContext

        def generate_row(image_path: Path):
            try:
                image_id = self.hub.get_image_id(image_path, self.subset)
                label = self.hub.get_image_label(image_path, self.subset)
            except KeyError as e:
                if self.ignore_missing_labels:
                    return None
                else:
                    raise e
            else:
                image = Image.open(image_path)

                if self.image_size:
                    image = image.resize(self.image_size)
                else:
                    current_min_length = min(*image.size)
                    current_max_length = max(*image.size)

                    # compute the minimum and maximum scale factors:

                    if self.min_length:
                        min_scale = float(self.min_length) / \
                                    float(current_min_length)
                    else:
                        min_scale = 0.

                    if self.max_length:
                        max_scale = float(self.max_length) / \
                                    float(current_max_length)
                    else:
                        max_scale = 1.

                    # scale at most to max_scale
                    # scale at least to min_scale (overrules max_scale)
                    scale = max(min_scale, min(max_scale, 1.))

                    if scale != 1.:
                        w, h = image.size
                        image = image.resize((round(w * scale),
                                              round(h * scale)))

                if image.mode != 'RGB':
                    image = image.convert('RGB')

                return {'image': np.asarray(image),
                        self.hub.image_id_field.name: image_id,
                        self.hub.label_field.name: label}

        schema = self._get_schema()

        img_stream = islice(self.images_dir.glob(self.glob), self.sample_size)
        rows_rdd = spark_ctx.parallelize(img_stream) \
            .repartition(self.num_partitions) \
            .map(generate_row) \
            .filter(lambda x: x is not None) \
            .map(lambda x: dict_to_spark_row(schema, x))

        df = self.write_cfg.session.createDataFrame(rows_rdd, schema.as_spark_schema())
        logging.info('Converted {} images into petastorm parquet store.'.format(df.count()))
        self.write_cfg.write_parquet(df, schema, self.output_url)


def main(convert_task: Type[ConvertTask]):
    """
    Convert images to a petastorm parquet store.
    Use one of the implementations in the submodules of this package.
    """
    parser = ArgumentParser(description='Convert images to a petastorm parquet store.')
    parser.add_arguments(convert_task, dest='convert_task')
    parser.add_arguments(LoggingConfig, dest='logging')
    args = parser.parse_args()
    args.convert_task.run()
