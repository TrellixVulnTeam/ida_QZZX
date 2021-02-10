from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Union, Optional, Tuple, Type

import numpy as np
from PIL import Image

from petastorm.unischema import Unischema, dict_to_spark_row
from simple_parsing import ArgumentParser

from simadv.common import LoggingConfig, ImageIdProvider
from simadv.io import Field, Schema, PetastormWriteConfig


@dataclass
class ConvertWriteConfig(PetastormWriteConfig):
    output_schema: Unischema = field(default=Schema.IMAGES, init=False)


@dataclass
class ConvertTask:

    images_dir: Union[str, Path]
    write_cfg: ConvertWriteConfig
    meta: ImageIdProvider

    ids_only: bool = False
    glob: str = '*.jpg'
    sample_size: Optional[int] = None
    subset: str = 'validation'
    image_size: Tuple[int, int] = (-1, -1)
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    num_partitions: int = 64 * 3

    def __post_init__(self):
        if isinstance(self.images_dir, str):
            self.images_dir = Path(self.images_dir)
        self.images_dir = self.images_dir.expanduser()
        assert self.images_dir.exists()

        assert self.subset in ['train', 'test', 'validation']

    def generate(self):
        img_stream = islice(self.images_dir.glob(self.glob), self.sample_size)
        for image_path in img_stream:
            image_id = self.meta.get_image_id(image_path, self.subset)
            image = Image.open(image_path)

            if self.image_size != (-1, -1):
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

            yield {Field.IMAGE.name: np.asarray(image),
                   Field.IMAGE_ID.name: image_id}

    def run(self):
        rows = [dict_to_spark_row(self.write_cfg.output_schema, row_dict) for row_dict in self.generate()]
        df = self.write_cfg.session.createDataFrame(rows, self.write_cfg.output_schema.as_spark_schema())
        self.write_cfg.write_parquet(df)


def main(convert_task: Type[ConvertTask]):
    """
    Convert images to a petastorm parquet store.
    Use one of the implementations in the submodules of this package.
    """
    parser = ArgumentParser(description='Convert images to a petastorm parquet store.')
    parser.add_arguments(convert_task, dest='convert_task')
    parser.add_arguments(LoggingConfig, dest='logging')
    args = parser.parse_args()
    args.convert_task.to_parquet()
