import abc
from dataclasses import dataclass, field

from simadv.describe.common import DictBasedImageDescriber
from simadv.torch import TorchImageClassifierSerialization, TorchConfig, TorchImageClassifier


@dataclass
class TorchImageDescriber(DictBasedImageDescriber, abc.ABC):

    classifier_serial: TorchImageClassifierSerialization
    torch_cfg: TorchConfig

    def __post_init__(self):
        # load the classifier from its name
        @dataclass
        class PartialTorchImageClassifier(TorchImageClassifier):
            torch_cfg: TorchConfig = field(default_factory=lambda: self.torch_cfg, init=False)

        self.classifier = PartialTorchImageClassifier.load(self.classifier_serial.path)
