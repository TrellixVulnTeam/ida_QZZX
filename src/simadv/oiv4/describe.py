from dataclasses import dataclass, field

from petastorm.unischema import UnischemaField

from simadv.oiv4.hub import OpenImagesV4Hub
from simadv.describe import main, DescribeTask


@dataclass
class OIV4DescribeTask(DescribeTask):
    id_field: UnischemaField = field(default=OpenImagesV4Hub.image_id_field, init=False)


if __name__ == '__main__':
    main(OIV4DescribeTask)
