import numpy as np
import torch

from liga.type2.attribution import SaliencyType2Explainer, IntegratedGradientsType2Explainer, \
    DeepLiftType2Explainer
from liga.type2.common import DummyType2Explainer
from liga.type2.lime import LimeType2Explainer


def test_dummy(image, interpreter, classifier):
    dummy = DummyType2Explainer(classifier=classifier,
                                interpreter=interpreter)
    assert list(dummy(image)) == [(1, 1.), (1, 1.), (0, 1.)]


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
