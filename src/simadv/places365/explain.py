from dataclasses import dataclass

from simadv.explain import TorchExplainTask, main
from simadv.places365.hub import Places365Hub


@dataclass
class Places365TorchExplainTask(TorchExplainTask):
    id_field = Places365Hub.image_id_field


if __name__ == '__main__':
    main(Places365TorchExplainTask)
