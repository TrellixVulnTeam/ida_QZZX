from unittest import mock

import numpy as np
import pytest

from ida.interpret.common import Interpreter
from ida.torch_extensions.classifier import TorchImageClassifier


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def image():
    yield np.ones((224, 224, 3), dtype=np.float) * 255.0


@pytest.fixture()
def classifier():
    yield TorchImageClassifier.from_json_file('places365_alexnet.json')


@pytest.fixture()
def interpreter(image, classifier):
    with mock.patch('ida.interpret.common.Interpreter') as interpreter:
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
        interpreter.concept_ids_to_counts = lambda x: Interpreter.concept_ids_to_counts(interpreter, x)
        yield interpreter
