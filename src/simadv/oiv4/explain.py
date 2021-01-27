from dataclasses import dataclass, field

from petastorm.unischema import UnischemaField

from simadv.explain import TorchExplainTask, main
from simadv.oiv4.hub import OpenImagesV4Hub


@dataclass
class OIV4TorchExplainTask(TorchExplainTask):
    id_field: UnischemaField = field(default=OpenImagesV4Hub.image_id_field, init=False)


if __name__ == '__main__':
    main(OIV4TorchExplainTask)
