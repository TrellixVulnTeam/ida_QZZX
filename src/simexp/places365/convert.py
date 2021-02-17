from dataclasses import dataclass, field

from simexp.common import ImageIdProvider
from simexp.convert import main, ConvertTask
from simexp.places365.metadata import Places365MetadataProvider


@dataclass
class Places365ConvertTask(ConvertTask):
    meta: ImageIdProvider = field(default_factory=Places365MetadataProvider, init=False)


if __name__ == '__main__':
    main(Places365ConvertTask)
