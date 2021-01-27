from dataclasses import dataclass

from simadv.oiv4.hub import OpenImagesV4Hub
from simadv.describe import main, DescribeTask


@dataclass
class OIV4DescribeTask(DescribeTask):
    id_field = OpenImagesV4Hub.image_id_field


if __name__ == '__main__':
    main(OIV4DescribeTask)
