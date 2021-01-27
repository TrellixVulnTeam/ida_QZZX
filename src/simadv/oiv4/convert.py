from dataclasses import dataclass, field

from simadv.convert import main, ConvertTask
from simadv.hub import SupervisedImageDataset
from simadv.oiv4.hub import OpenImagesV4Hub


@dataclass
class OIV4ConvertTask(ConvertTask):
    hub: SupervisedImageDataset = field(default=OpenImagesV4Hub, init=False)


if __name__ == '__main__':
    main(OIV4ConvertTask)
