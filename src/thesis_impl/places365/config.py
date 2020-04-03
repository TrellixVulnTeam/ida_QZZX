import logging

import torch
from typeguard import typechecked


class LoggingConfig:

    @staticmethod
    def set_from_args(log_args):
        if log_args.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    @staticmethod
    def setup_parser(parser, default_cache_dir='~/.cache/places-365'):
        """
        Adds an argument to `parser to specify a `cache_dir`.
        """
        parser.add_argument('--debug', action='store_true',
                            help='whether to output more information '
                                 'for debugging')


class WebCacheConfig:

    @staticmethod
    def from_args(cache_args):
        return cache_args.cache_dir

    @staticmethod
    def setup_parser(parser, default_cache_dir='~/.cache/places-365'):
        """
        Adds an argument to `parser to specify a `cache_dir`.
        """
        parser.add_argument('--cache-dir', type=str, default=default_cache_dir,
                            help='where to cache files downloaded from the web')


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
                 spark_exec_memory: str, num_partitions: int,
                 row_group_size_mb: int):
        self.spark_master = spark_master
        self.spark_driver_memory = spark_driver_memory
        self.spark_exec_memory = spark_exec_memory
        self.num_partitions = num_partitions
        self.row_group_size_mb = row_group_size_mb

    @staticmethod
    def from_args(write_args):
        return PetastormWriteConfig(write_args.spark_master,
                                    write_args.spark_driver_memory,
                                    write_args.spark_exec_memory,
                                    write_args.num_partitions,
                                    write_args.row_size)

    @staticmethod
    def setup_parser(parser, default_spark_master='local[8]',
                     default_spark_driver_memory='40g',
                     default_spark_exec_memory='2g',
                     default_num_partitions=10,
                     default_row_size=1024):
        """
        Adds all arguments to `parser that are necessary to construct
        a `PetastormWriteConfig`.
        """
        parser.add_argument('--spark-master', type=str,
                            default=default_spark_master)
        parser.add_argument('--spark-driver-memory', type=str,
                            default=default_spark_driver_memory)
        parser.add_argument('--spark-exec-memory', type=str,
                            default=default_spark_exec_memory)
        parser.add_argument('--num-partitions', type=int,
                            default=default_num_partitions)
        parser.add_argument('--row-size', type=int,
                            default=default_row_size)
