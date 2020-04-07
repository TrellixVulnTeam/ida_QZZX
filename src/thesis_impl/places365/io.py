import torch
from petastorm import make_reader
from petastorm.pytorch import DataLoader

import thesis_impl.places365.config as cfg


def peta_loader(data_url, schema_fields,
                read_cfg: cfg.PetastormReadConfig):
    reader = make_reader(data_url,
                         schema_fields=schema_fields,
                         shuffle_row_groups=read_cfg.shuffle_row_groups,
                         reader_pool_type=read_cfg.reader_pool_type,
                         workers_count=read_cfg.reader_workers_count)
    return DataLoader(reader, batch_size=read_cfg.batch_size)


def _get_image_tensors(batch, device: torch.device):
    images = batch['image'].to(device, torch.float)
    # pytorch expects channels before width/height
    images = images.permute(0, 3, 1, 2).div(255)
    assert images.min().item() >= 0
    assert images.max().item() <= 1
    return images


def _get_label_tensors(batch, device: torch.device):
    return batch['label_id'].to(device, torch.int16)


def unsupervised_loader(data_url, read_cfg: cfg.PetastormReadConfig,
                        device: torch.device):
    for batch in peta_loader(data_url, ['image'], read_cfg):
        yield _get_image_tensors(batch, device)


def supervised_loader(data_url, read_cfg: cfg.PetastormReadConfig,
                      device: torch.device):
    for batch in peta_loader(data_url, ['image', 'label_id'], read_cfg):
        yield _get_image_tensors(batch, device),\
              _get_label_tensors(batch, device)
