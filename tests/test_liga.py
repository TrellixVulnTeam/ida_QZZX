from liga.liga import random_subsets


def test_that_random_subsets_runs(rng, interpreter):
    concept_ids = [4, 1, 1, 5]
    influences = [-10., 2., 0., 3.]

    subsets = list(random_subsets(rng=rng,
                                  concept_ids=concept_ids,
                                  influences=influences,
                                  interpreter=interpreter))
    assert len(subsets) > 0  # each concept instance with negative or zero influence must be removed at least once


def test_random_subsets_with_dummy_type2(rng):
    concept_ids = [4, 1, 1, 5]
    influences = [5., 2., 1., 1.]
    num_concepts = 6

    subsets = list(random_subsets(rng=rng,
                                  concept_ids=concept_ids,
                                  influences=influences))
    assert subsets == []
