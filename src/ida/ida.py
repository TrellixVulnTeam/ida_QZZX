import abc
import time
from typing import Iterable, Tuple, Dict, Any, Optional, List

import numpy as np
from sklearn.pipeline import Pipeline

from ida.interpret.common import Interpreter
from ida.torch_extensions.classifier import TorchImageClassifier
from ida.type1.common import Type1Explainer
from ida.type2.common import Type2Explainer
from ida.common import NestedLogger
from ida.util.itertools import random_sublists


class Decorrelator(abc.ABC):

    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter):
        self.classifier = classifier
        self.interpreter = interpreter

    @property
    def stats(self):
        return {}

    def calibrate(self, images_iter: Iterable[Tuple[str, np.ndarray]]):
        pass

    @abc.abstractmethod
    def __call__(self,
                 rng: np.random.Generator,
                 image: Optional[np.ndarray] = None,
                 image_id: Optional[str] = None) -> Iterable[Tuple[List[int], int]]:
        pass


class NoDecorrelator(Decorrelator):

    def __call__(self,
                 rng: np.random.Generator,
                 image: Optional[np.ndarray] = None,
                 image_id: Optional[str] = None,
                 **kwargs) -> Iterable[Tuple[List[int], int]]:
        concept_ids, _ = list(zip(*self.interpreter(image=image,
                                                    image_id=image_id)))
        counts = self.interpreter.concept_ids_to_counts(concept_ids)
        predicted_class = self.classifier.predict_single(image=image, image_id=image_id)
        yield counts, predicted_class

    def __str__(self):
        return 'NoDecorrelator'


class Type2Decorrelator(Decorrelator):

    def __init__(self,
                 type2: Type2Explainer,
                 max_concept_area: float,
                 max_concept_overlap: float):
        super().__init__(classifier=type2.classifier,
                         interpreter=type2.interpreter)
        self.type2 = type2
        self.max_concept_area = max_concept_area
        self.max_concept_overlap = max_concept_overlap

    @property
    def stats(self):
        return self.type2.stats

    def calibrate(self, images_iter: Iterable[Tuple[str, np.ndarray]]):
        self.type2.calibrate(images_iter=images_iter)

    def __call__(self,
                 rng: np.random.Generator,
                 image: Optional[np.ndarray] = None,
                 image_id: Optional[str] = None) -> Iterable[Tuple[List[int], int]]:
        concept_influences = list(self.type2(image=image,
                                             image_id=image_id))
        if len(concept_influences) > 0:
            concept_ids, masks, influences = list(zip(*concept_influences))
            counts = self.interpreter.concept_ids_to_counts(concept_ids)
            predicted_class = self.classifier.predict_single(image=image, image_id=image_id)
            yield counts, predicted_class

            num_pixels = float(image.shape[0] * image.shape[1])
            to_be_dropped = []
            for c, m, i in concept_influences:
                if i <= 0 and float(np.count_nonzero(m)) / num_pixels <= self.max_concept_area:
                    min_overlap = self.max_concept_overlap
                    overlapping = self.interpreter.get_overlapping_concepts(mask=m,
                                                                            all_ids_and_masks=zip(concept_ids, masks),
                                                                            is_overlap_at=min_overlap)
                    overlaps_other_influential_concept = any(influences[concept_ids.index(overlapping_concept_id)] > 0
                                                             for overlapping_concept_id, _ in overlapping)
                    if not overlaps_other_influential_concept:
                        to_be_dropped.append(c)

            to_be_dropped_counts = np.array(self.interpreter.concept_ids_to_counts(to_be_dropped))
            sublist_iter = random_sublists(rng=rng,
                                           the_list=to_be_dropped,
                                           max_size=None,
                                           include_empty=False)

            while np.any(to_be_dropped_counts > 0):
                dropped_subset = next(sublist_iter)
                dropped_counts = np.array(self.interpreter.concept_ids_to_counts(dropped_subset))
                to_be_dropped_counts -= dropped_counts
                yield (counts - dropped_counts).tolist(), predicted_class

    def __str__(self):
        return 'Type2Decorrelator(type2={}, max_concept_area={})'.format(str(self.type2), self.max_concept_area)


class CounterfactualDecorrelator(Decorrelator):

    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter,
                 max_perturbed_area: float,
                 max_concept_overlap: float):
        super().__init__(classifier=classifier,
                         interpreter=interpreter)
        self.max_perturbed_area = max_perturbed_area
        self.max_concept_overlap = max_concept_overlap

    def __call__(self,
                 rng: np.random.Generator,
                 image: Optional[np.ndarray] = None,
                 image_id: Optional[str] = None,
                 **kwargs) -> Iterable[Tuple[List[int], int]]:
        ids_and_masks = list(self.interpreter(image=image,
                                              image_id=image_id,
                                              **kwargs))
        if len(ids_and_masks) > 0:
            ids, masks = list(zip(*ids_and_masks))
            counts = self.interpreter.concept_ids_to_counts(ids)
            predicted_class = self.classifier.predict_single(image=image, image_id=image_id)
            yield counts, predicted_class

            cf_iter = self.interpreter.get_counterfactuals(image=image,
                                                           image_id=image_id,
                                                           max_perturbed_area=self.max_perturbed_area,
                                                           max_concept_overlap=self.max_concept_overlap)
            for cf_counts, cf_image in cf_iter:
                perturbed_true_class = self.classifier.predict_single(image=cf_image)
                yield cf_counts, perturbed_true_class

    def __str__(self):
        return ('CounterfactualDecorrelator(max_perturbed_area={}, max_concept_overlap={})'
                .format(self.max_perturbed_area, self.max_concept_overlap))


def ida(rng: np.random.Generator,
        type1: Type1Explainer,
        image_iter: Iterable[Tuple[str, np.ndarray]],
        decorrelator: Decorrelator,
        log_nesting: int = 0,
        log_frequency_s: int = 20,
        **kwargs) -> Tuple[Pipeline, Dict[str, Any]]:

    logger = NestedLogger()
    logger.log_nesting = log_nesting
    with logger.log_task('Running LIGA...'):
        concept_counts = []
        predicted_classes = []
        augmentation_count = 0
        last_log_time = time.time()

        for obs_no, (image_id, image) in enumerate(image_iter):
            augmentation_count -= 1  # do not count the original image
            for counts, predicted_class in decorrelator(rng, image, image_id):
                concept_counts.append(counts)
                predicted_classes.append(predicted_class)
                augmentation_count += 1

            current_time = time.time()
            if current_time - last_log_time > log_frequency_s:
                with logger.log_task('Status update'):
                    logger.log_item('Processed {} observations'
                                    .format(obs_no + 1))
                    logger.log_item('LIGA\'s augmentation produced {} additional observations.'
                                    .format(augmentation_count))
                last_log_time = current_time

        stats = {'augmentation_count': augmentation_count}
        stats.update(decorrelator.stats)

        with logger.log_task('Fitting surrogate model...'):
            surrogate = type1(concept_counts=concept_counts,
                              predicted_classes=predicted_classes,
                              all_classes=list(range(decorrelator.classifier.num_classes)),
                              **kwargs)
        return surrogate, stats
