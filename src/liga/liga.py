import time
from typing import Iterable, Tuple, TypeVar, Dict, Any

import numpy as np

from liga.type1.common import Type1Explainer
from liga.type2.common import Type2Explainer
from simexp.common import NestedLogger


T = TypeVar('T')


def liga(rng: np.random.Generator,
         type1: Type1Explainer[T],
         type2: Type2Explainer,
         image_iter: Iterable[Tuple[str, np.ndarray]],
         log_nesting: int = 0,
         log_frequency_s: int = 10,
         **kwargs) -> Tuple[T, Dict[str, Any]]:

    logger = NestedLogger()
    logger.log_nesting = log_nesting
    with logger.log_task('Running LIGA...'):
        num_concepts = len(type2.interpreter.concepts)
        concept_counts = []
        predicted_classes = []
        influential_count = 0
        augmentation_count = 0
        last_log_time = time.time()

        for obs_no, (image_id, image) in enumerate(image_iter):
            concept_influences = list(type2(image=image,
                                            image_id=image_id,
                                            **kwargs))
            if len(concept_influences) == 0:
                continue

            concept_ids, influences = list(zip(*concept_influences))
            counts = concept_ids_to_counts(concept_ids,
                                           num_concepts)
            if not any([i > 0 for i in influences]):
                continue

            influential_count += 1
            concept_counts.append(counts)
            predicted_class = type2.classifier.predict_single(image)
            predicted_classes.append(predicted_class)

            for counts_subset in random_subsets(rng=rng,
                                                concept_ids=concept_ids,
                                                influences=influences,
                                                num_concepts=num_concepts):
                augmentation_count += 1
                concept_counts.append(counts_subset)
                predicted_classes.append(predicted_class)

            current_time = time.time()
            if current_time - last_log_time > log_frequency_s:
                with logger.log_task('Status update'):
                    logger.log_item('Processed {} observations'
                                    .format(obs_no + 1))
                    logger.log_item('{} observations had influential concepts'
                                    .format(influential_count))
                    logger.log_item('LIGA\'s augmentation produced {} additional observations.'
                                    .format(augmentation_count))
                last_log_time = current_time

        stats = {'influential_count': influential_count,
                 'augmentation_count': augmentation_count}
        surrogate = type1(concept_counts,
                          predicted_classes,
                          list(range(type2.classifier.num_classes)),
                          **kwargs)
        return surrogate, stats


def concept_ids_to_counts(concept_ids: Iterable[int],
                          num_concepts: int) -> [int]:
    counts = [0] * num_concepts
    for concept_id in concept_ids:
        counts[concept_id] += 1
    return counts


def random_subsets(rng: np.random.Generator,
                   concept_ids: [int],
                   influences: [float],
                   num_concepts: int):
    concept_ids = np.array(concept_ids)
    influences = np.array(influences)

    droppable_idx = influences <= 0
    never_dropped = np.ones(np.count_nonzero(droppable_idx),
                            dtype=np.bool_)

    while np.any(never_dropped):
        concept_ids_idx = np.ones_like(concept_ids, dtype=np.bool_)
        random_flip = rng.random(never_dropped.shape) > .5
        concept_ids_idx[droppable_idx] = random_flip
        never_dropped = np.bitwise_and(never_dropped, random_flip)
        selected_concept_ids = concept_ids[concept_ids_idx]
        yield concept_ids_to_counts(concept_ids=selected_concept_ids,
                                    num_concepts=num_concepts)
