from dataclasses import dataclass, field

from petastorm.unischema import UnischemaField

from simadv.explain import TorchExplainTask, main
from simadv.places365.hub import Places365Hub


@dataclass
class Places365TorchExplainTask(TorchExplainTask):
    id_field: UnischemaField = field(default=Places365Hub.image_id_field, init=False)


if __name__ == '__main__':
    main(Places365TorchExplainTask)
