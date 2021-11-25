from dataclasses import dataclass, field

from ida.common import ImageIdProvider
from ida.convert import main, ConvertTask
from ida.places365.metadata import Places365MetadataProvider


@dataclass
class Places365ConvertTask(ConvertTask):
    meta: ImageIdProvider = field(default_factory=Places365MetadataProvider, init=False)


if __name__ == '__main__':
    main(Places365ConvertTask)
