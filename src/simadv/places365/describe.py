from dataclasses import dataclass, field

from petastorm.unischema import UnischemaField

from simadv.places365.hub import Places365Hub
from simadv.describe import main, DescribeTask


@dataclass
class Places365DescribeTask(DescribeTask):
    id_field: UnischemaField = field(default=Places365Hub.image_id_field, init=False)


if __name__ == '__main__':
    main(Places365DescribeTask)
