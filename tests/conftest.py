from unittest import mock

import numpy as np
import pytest

from liga.torch_extensions.classifier import TorchImageClassifierSerialization, TorchImageClassifierLoader
from simexp.describe.torch_based.common import TorchConfig


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def image():
    yield np.ones((224, 224, 3), dtype=np.float) * 255.0


@pytest.fixture()
def classifier():
    torch_cfg = TorchConfig()
    alexnet_serial = TorchImageClassifierSerialization('places365_alexnet.json')
    yield TorchImageClassifierLoader(alexnet_serial, torch_cfg).classifier


@pytest.fixture()
def interpreter(image, classifier):
    with mock.patch('liga.interpret.common.Interpreter') as interpreter:
        interpreter.concepts = ['teacup', 'banana']
        interpreter.num_concepts = 2
        teacup_mask = np.zeros(image.shape[:2], dtype=np.bool_)
        teacup_mask[20:30, 50:100] = True
        banana_mask = np.zeros(image.shape[:2], dtype=np.bool_)
        banana_mask[25:40, 80:190] = True
        banana_mask_2 = np.zeros(image.shape[:2], dtype=np.bool_)
        banana_mask_2[30:45, 100:210] = True
        interpreter.return_value = [(1, banana_mask),
                                    (1, banana_mask),
                                    (0, teacup_mask)]
        yield interpreter
