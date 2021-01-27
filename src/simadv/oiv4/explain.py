from dataclasses import dataclass

from simadv.explain import TorchExplainTask, main
from simadv.oiv4.hub import OpenImagesV4Hub


@dataclass
class OIV4TorchExplainTask(TorchExplainTask):
    id_field = OpenImagesV4Hub.image_id_field


if __name__ == '__main__':
    main(OIV4TorchExplainTask)
