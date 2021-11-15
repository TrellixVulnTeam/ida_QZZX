import abc
from dataclasses import dataclass
from typing import Iterable

import torch

from liga.common import RowDict
from simexp.describe.common import DictBasedImageDescriber


@dataclass
class TorchConfig:
    use_cuda: bool = True  # whether to use CUDA if available

    def __post_init__(self):
        self.use_cuda = self.use_cuda and torch.cuda.is_available()
        self.device: torch.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')


@dataclass
class BatchedTorchImageDescriber(DictBasedImageDescriber, abc.ABC):
    """
    Reads batches of image data from a petastorm store
    and converts these images to Torch tensors.

    Attention: If the input images can have different sizes, you *must*
    set `read_cfg.batch_size` to 1!

    Subclasses can describe the produced tensors as other data
    batch-wise by implementing the `describe_batch` method.
    """

    torch_cfg: TorchConfig
    batch_size: int

    def batch_iter(self):
        def to_tensor(batch_columns):
            row_ids, image_arrays = batch_columns
            image_tensors = torch.as_tensor(image_arrays) \
                .to(self.torch_cfg.device, torch.float) \
                .div(255)

            return row_ids, image_tensors

        current_batch = []

        for row in self.read_cfg.make_reader(None):
            current_batch.append((row.image_id, row.image))

            if len(current_batch) < self.batch_size:
                continue

            yield to_tensor(list(zip(*current_batch)))
            current_batch.clear()

        if current_batch:
            yield to_tensor(list(zip(*current_batch)))

    @abc.abstractmethod
    def describe_batch(self, ids, batch) -> Iterable[RowDict]:
        pass

    def clean_up(self):
        """
        Can be overridden to clean up more stuff.
        """
        torch.cuda.empty_cache()  # release all memory that can be released

    def generate(self):
        with torch.no_grad():
            for ids, batch in self.batch_iter():
                yield from self.describe_batch(ids, batch)

        self.clean_up()
