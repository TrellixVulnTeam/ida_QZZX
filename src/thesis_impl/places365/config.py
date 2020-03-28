import torch
from typeguard import typechecked


class TorchConfig:

    @typechecked
    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = self.device.type == 'cuda'

    def set_device(self):
        """
        Context manager that executes the contained code with the
        configured torch device.
        """
        return torch.cuda.device(self.device if self.use_cuda else None)

    @staticmethod
    def from_args(torch_args):
        if not torch_args.no_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return TorchConfig(device=device)

    @staticmethod
    def setup_parser(parser):
        """
        Adds all arguments to `parser that are necessary to construct
        a `TorchConfig`.
        """
        parser.add_argument('--no-cuda', action='store_true',
                            help='Disable CUDA')


class PetastormReadConfig:

    @typechecked
    def __init__(self, batch_size: int, shuffle_row_groups: bool):
        self.batch_size = batch_size
        self.shuffle_row_groups = shuffle_row_groups

    @staticmethod
    def from_args(read_args):
        return PetastormReadConfig(read_args.batch_size,
                                   not read_args.no_shuffle)

    @staticmethod
    def setup_parser(parser, default_batch_size=512):
        """
        Adds all arguments to `parser that are necessary to construct
        a `PetastormReadConfig`.
        """
        parser.add_argument('--batch-size', type=int,
                            default=default_batch_size)
        parser.add_argument('--no-shuffle', action='store_true')


class PetastormWriteConfig:

    @typechecked
    def __init__(self, spark_master: str, spark_driver_memory: str,
                 row_group_size_mb: int):
        self.spark_master = spark_master
        self.spark_driver_memory = spark_driver_memory
        self.row_group_size_mb = row_group_size_mb

    @staticmethod
    def from_args(write_args):
        return PetastormReadConfig(write_args.spark_master,
                                   write_args.spark_memory,
                                   write_args.row_size)

    @staticmethod
    def setup_parser(parser, default_spark_master='local[8]',
                     default_spark_memory='8g', default_row_size=1024):
        """
        Adds all arguments to `parser that are necessary to construct
        a `PetastormWriteConfig`.
        """
        parser.add_argument('--spark-master', type=str,
                            default=default_spark_master)
        parser.add_argument('--spark-memory', type=str,
                            default=default_spark_memory)
        parser.add_argument('--row-size', type=int,
                            default=default_row_size)
