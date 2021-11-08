from unittest import mock

import numpy as np
import pytest
import torch

from liga.type2.attribution import SaliencyType2Explainer, IntegratedGradientsType2Explainer, \
    DeepLiftType2Explainer
from liga.type2.lime import LimeType2Explainer
from simexp.describe.torch_based.common import TorchConfig
from liga.torch_extensions.classifier import TorchImageClassifierSerialization, TorchImageClassifierLoader


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


class TestLime:

    def test_from_interp_rep_transform(self):
        image = np.array([
            [[1., 2., 1.], [1., 2., 1.]],
            [[1., 1., 2.], [1., 1., 2.]]
        ])

        concept_mask_1 = np.array([
            [0, 1],
            [0, 0]
        ], dtype=np.bool_)  # selects only top right pixel

        concept_mask_2 = np.array([
            [0, 0],
            [1, 1]
        ], dtype=np.bool_)  # selects bottom row of pixels

        concept_masks = [concept_mask_1, concept_mask_2]

        concept_bits_1 = torch.tensor([1, 0])  # this should mask all pixels in concept_mask_2 with the baseline value
        result_1 = LimeType2Explainer.from_interp_rep_transform(concept_bits=concept_bits_1,
                                                                _=torch.tensor(0),
                                                                image=image,
                                                                concept_masks=concept_masks,
                                                                baseline_rgb=(0., 0., 0.))
        assert result_1.equal(torch.tensor([[
            [  # channel r
                [1., 1.],
                [0., 0.]
            ],
            [  # channel g
                [2., 2.],
                [0., 0.]
            ],
            [  # channel b
                [1., 1.],
                [0., 0.]]
        ]]))

        concept_bits_2 = torch.tensor([0, 0])
        result_2 = LimeType2Explainer.from_interp_rep_transform(concept_bits=concept_bits_2,
                                                                _=torch.tensor(0),
                                                                image=image,
                                                                concept_masks=concept_masks,
                                                                baseline_rgb=(0., 0., 0.))
        assert result_2.equal(torch.tensor([[
            [  # channel r
                [1., 0.],
                [0., 0.]
            ], [  # channel g
                [2., 0.],
                [0., 0.]
            ], [  # channel b
                [1., 0.],
                [0., 0.]
            ]
        ]]))

    def test_that_lime_runs(self, image, interpreter, classifier):
        lime = LimeType2Explainer(classifier=classifier,
                                  interpreter=interpreter)
        lime(image=image,
             image_id='test_img')


class TestGradient:

    def test_that_saliency_runs(self, image, interpreter, classifier):
        saliency = SaliencyType2Explainer(classifier=classifier,
                                          interpreter=interpreter)
        saliency(image=image,
                 image_id='test_img')

    def test_that_igrad_runs(self, image, interpreter, classifier):
        saliency = IntegratedGradientsType2Explainer(classifier=classifier,
                                                     interpreter=interpreter)
        saliency(image=image,
                 image_id='test_img')

    def test_that_deeplift_runs(self, image, interpreter, classifier):
        saliency = DeepLiftType2Explainer(classifier=classifier,
                                          interpreter=interpreter)
        saliency(image=image,
                 image_id='test_img')
