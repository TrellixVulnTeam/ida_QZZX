from dataclasses import dataclass, field
from functools import partial, reduce
from typing import List, Type, Any

from pyspark.sql import DataFrame
from simple_parsing import ArgumentParser
from simadv.common import LoggingConfig
from simadv.describe.colors import PerceivableColorsImageDescriber
from simadv.describe.common import ImageDescriber, ImageReadConfig
from simadv.describe.tf_based.objects import OIV4ObjectsImageDescriber
from simadv.describe.torch_based.objects import CocoObjectsImageDescriber
from simadv.io import Field, PetastormWriteConfig


def _partial_describer(cls: Type[Any], read_cfg: ImageReadConfig, write_cfg: PetastormWriteConfig):

    @dataclass
    class PartialDescriber(cls):
        read_cfg: ImageReadConfig = field(default_factory=lambda: read_cfg, init=False)
        write_cfg: PetastormWriteConfig = field(default_factory=lambda: write_cfg, init=False)

    return PartialDescriber


@dataclass
class DescriberConfig:
    read_cfg: ImageReadConfig
    write_cfg: PetastormWriteConfig

    # list of abbreviations for the describers to use
    select_by_abbreviation: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.describer_types = {'colors': PerceivableColorsImageDescriber,
                                'coco_objects': CocoObjectsImageDescriber,
                                'oiv4_objects': OIV4ObjectsImageDescriber}
        if self.select_by_abbreviation:
            self.describer_types = {k: v for (k, v) in self.describer_types.items()
                                    if k in self.select_by_abbreviation}


@dataclass
class JoinedImageDescriber(ImageDescriber):
    """
    Joins the results of multiple image describers.
    """

    describers: [ImageDescriber]

    def to_df(self):
        dfs = [d.to_df() for d in self.describers]
        # use outer join in case there are missing values
        op = partial(DataFrame.join, on=Field.IMAGE_ID.name, how='outer')
        return reduce(op, dfs)


def main():
    """
    Describe images with abstract and familiar attributes.
    The images are read from a petastorm parquet store. In this parquet store,
    there must be two fields:

      - A field `image_id` holding a unique id for each image.
      - A field `image` holding image data encoded with the petastorm png encoder.

    Use one of the implementations in the submodules of this package.
    """
    parser = ArgumentParser(description='Describe images with abstract and familiar concepts.')
    parser.add_arguments(DescriberConfig, 'describer_config')
    parser.add_arguments(LoggingConfig, 'logging')
    general_args, remaining = parser.parse_known_args()

    describers = []
    for describer_type in general_args.describer_config.describer_types.values():
        parser = ArgumentParser()
        parser.add_arguments(_partial_describer(describer_type, general_args.describer_config.read_cfg,
                                                general_args.describer_config.write_cfg), 'describer')
        describer_args, remaining = parser.parse_known_args(remaining)
        describers.append(describer_args.describer)

    joiner = JoinedImageDescriber()


if __name__ == '__main__':
    main()
