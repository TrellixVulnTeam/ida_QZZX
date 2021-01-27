from dataclasses import dataclass

from simadv.places365.hub import Places365Hub
from simadv.describe import main, DescribeTask


@dataclass
class Places365DescribeTask(DescribeTask):
    id_field = Places365Hub.image_id_field


if __name__ == '__main__':
    main(Places365DescribeTask)
