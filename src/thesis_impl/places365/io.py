import torch
from petastorm import make_reader
from petastorm.pytorch import DataLoader

import thesis_impl.places365.config as cfg


def peta_loader(data_url, schema_fields,
                batch_size=cfg.Petastorm.Read.batch_size,
                shuffle_row_groups=cfg.Petastorm.Read.shuffle_row_groups,
                **kwargs):
    reader = make_reader(data_url,
                         schema_fields=schema_fields,
                         shuffle_row_groups=shuffle_row_groups,
                         **kwargs)
    return DataLoader(reader, batch_size=batch_size)


def _get_image_tensors(batch, device):
    images = batch['image'].to(device, torch.float)
    # pytorch expects channels before width/height
    images = images.permute(0, 3, 1, 2).div(255)
    assert images.min().item() >= 0
    assert images.max().item() <= 1
    return images


def _get_label_tensors(batch, device):
    return batch['label_id'].to(device, torch.int16)


def unsupervised_loader(data_url, device=cfg.Torch.DEFAULT_DEVICE, **kwargs):
    for batch in peta_loader(data_url, ['image'], **kwargs):
        yield _get_image_tensors(batch, device)


def supervised_loader(data_url, device=cfg.Torch.DEFAULT_DEVICE, **kwargs):
    for batch in peta_loader(data_url, ['image', 'label_id'], **kwargs):
        yield _get_image_tensors(batch, device),\
              _get_label_tensors(batch, device)
