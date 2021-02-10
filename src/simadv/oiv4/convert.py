from dataclasses import dataclass, field

from simadv.common import ImageIdProvider
from simadv.convert import main, ConvertTask
from simadv.oiv4.metadata import OIV4MetadataProvider


@dataclass
class OIV4ConvertTask(ConvertTask):
    meta: ImageIdProvider = field(default_factory=OIV4MetadataProvider, init=False)


if __name__ == '__main__':
    main(OIV4ConvertTask)
