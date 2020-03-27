import torch


class Torch:
    DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')


class Petastorm:
    class Read:
        batch_size = 512
        shuffle_row_groups = True

    class Write:
        row_group_size_mb = 1024,
        spark_driver_memory = '8g'
        spark_master = 'local[8]'
