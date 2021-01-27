from dataclasses import dataclass, field

from simadv.convert import main, ConvertTask
from simadv.hub import SupervisedImageDataset
from simadv.places365.hub import Places365Hub


@dataclass
class Places365ConvertTask(ConvertTask):
    hub: SupervisedImageDataset = field(default_factory=Places365Hub, init=False)


if __name__ == '__main__':
    main(Places365ConvertTask)
