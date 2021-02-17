import abc
from dataclasses import dataclass, field

from petastorm.unischema import Unischema

from simadv.spark import Schema, PetastormReadConfig, DictBasedDataGenerator


@dataclass
class ImageReadConfig(PetastormReadConfig):
    input_schema: Unischema = field(default=Schema.IMAGES, init=False)


@dataclass
class DictBasedImageDescriber(DictBasedDataGenerator, abc.ABC):

    # how to read input images
    read_cfg: ImageReadConfig

    output_schema: Unischema = field(default=Schema.CONCEPT_MASKS, init=False)
