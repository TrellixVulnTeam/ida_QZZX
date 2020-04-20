from petastorm import make_reader
from petastorm.pytorch import DataLoader

import thesis_impl.config as cfg


def torch_peta_loader(data_url, read_cfg: cfg.PetastormReadConfig,
                      schema_fields=None):
    reader = make_reader(data_url,
                         shuffle_row_groups=read_cfg.shuffle_row_groups,
                         reader_pool_type=read_cfg.pool_type,
                         workers_count=read_cfg.workers_count,
                         schema_fields=schema_fields)
    return DataLoader(reader, batch_size=read_cfg.batch_size)
