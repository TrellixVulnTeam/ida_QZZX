import abc
import time
from typing import Iterable, Tuple, TypeVar, Dict, Any, Optional, List

import numpy as np

from liga.interpret.common import Interpreter
from liga.torch_extensions.classifier import TorchImageClassifier
from liga.type1.common import Type1Explainer
from liga.type2.common import Type2Explainer
from liga.common import NestedLogger
from liga.util.itertools import random_sublists


class Resampler(abc.ABC):

    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter):
        self.classifier = classifier
        self.interpreter = interpreter

    @abc.abstractmethod
    def __call__(self,
                 rng: np.random.Generator,
                 image: Optional[np.ndarray] = None,
                 image_id: Optional[str] = None,
                 **kwargs) -> Iterable[Tuple[List[int], int]]:
        pass


class DummyResampler(Resampler):

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
        return 'none'


class Type2Resampler(Resampler):

    def __init__(self,
                 type2: Type2Explainer,
                 max_concept_area: float):
        super().__init__(classifier=type2.classifier,
                         interpreter=type2.interpreter)
        self.type2 = type2
        self.max_concept_area = max_concept_area

    def __call__(self,
                 rng: np.random.Generator,
                 image: Optional[np.ndarray] = None,
                 image_id: Optional[str] = None,
                 **kwargs) -> Iterable[Tuple[List[int], int]]:
        concept_influences = list(self.type2(image=image,
                                             image_id=image_id,
                                             **kwargs))
        if len(concept_influences) != 0:
            concept_ids, masks, influences = list(zip(*concept_influences))
            counts = self.interpreter.concept_ids_to_counts(concept_ids)
            predicted_class = self.classifier.predict_single(image=image, image_id=image_id)
            yield counts, predicted_class

            num_pixels = float(image.shape[0] * image.shape[1])
            to_be_dropped = [c for c, m, i in concept_influences
                             if i <= 0 and float(np.count_nonzero(m)) / num_pixels <= self.max_concept_area]
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
        return 'type2({})'.format(str(self.type2))


class RandomCounterfactualResampler(Resampler):

    def __init__(self,
                 classifier: TorchImageClassifier,
                 interpreter: Interpreter,
                 num_counterfactuals: int,
                 max_perturbed_area: float):
        super().__init__(classifier=classifier,
                         interpreter=interpreter)
        self.num_counterfactuals = num_counterfactuals
        self.max_perturbed_area = max_perturbed_area

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

            cf_iter = self.interpreter.get_counterfactuals(rng,
                                                           image=image,
                                                           image_id=image_id,
                                                           max_counterfactuals=self.num_counterfactuals,
                                                           max_perturbed_concepts=1,
                                                           max_perturbed_area=self.max_perturbed_area)
            for cf_counts, cf_image in cf_iter:
                perturbed_true_class = self.classifier.predict_single(image=cf_image)
                yield cf_counts, perturbed_true_class

    def __str__(self):
        return 'random_counterfactual'


T = TypeVar('T')


def liga(rng: np.random.Generator,
         type1: Type1Explainer[T],
         image_iter: Iterable[Tuple[str, np.ndarray]],
         resampler: Resampler,
         log_nesting: int = 0,
         log_frequency_s: int = 20,
         **kwargs) -> Tuple[T, Dict[str, Any]]:

    logger = NestedLogger()
    logger.log_nesting = log_nesting
    with logger.log_task('Running LIGA...'):
        concept_counts = []
        predicted_classes = []
        augmentation_count = 0
        last_log_time = time.time()

        for obs_no, (image_id, image) in enumerate(image_iter):
            augmentation_count -= 1  # do not count the original image
            for counts, predicted_class in resampler(rng, image, image_id):
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

        with logger.log_task('Fitting surrogate model...'):
            surrogate = type1(concept_counts=concept_counts,
                              predicted_classes=predicted_classes,
                              all_classes=list(range(resampler.classifier.num_classes)),
                              **kwargs)
        return surrogate, stats
