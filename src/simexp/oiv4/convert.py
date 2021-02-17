from dataclasses import dataclass, field

from simexp.common import ImageIdProvider
from simexp.convert import main, ConvertTask
from simexp.oiv4.metadata import OIV4MetadataProvider


@dataclass
class OIV4ConvertTask(ConvertTask):
    meta: ImageIdProvider = field(default_factory=OIV4MetadataProvider, init=False)


if __name__ == '__main__':
    main(OIV4ConvertTask)
