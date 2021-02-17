from dataclasses import dataclass, field

from petastorm.unischema import Unischema

from simadv.spark import Schema, PetastormReadConfig


@dataclass
class ImageReadConfig(PetastormReadConfig):
    input_schema: Unischema = field(default=Schema.IMAGES, init=False)
