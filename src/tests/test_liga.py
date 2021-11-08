import numpy as np
from pytest import fixture

from liga.liga import random_subsets


@fixture
def rng():
    return np.random.default_rng(42)


def test_that_random_subsets_runs(rng):
    concept_ids = [4, 1, 1, 5]
    influences = [-10., 2., -5., 3.]
    num_concepts = 6

    random_subsets(rng,
                   concept_ids,
                   influences,
                   num_concepts)

