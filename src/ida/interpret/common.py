import abc
from functools import reduce
from typing import Optional, Tuple, Iterable, List

import numpy as np


class Interpreter(abc.ABC):

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    @property
    @abc.abstractmethod
    def concepts(self) -> [str]:
        pass

    def get_implied_concepts(self, concept_id: int) -> Iterable[int]:
        """
        Subclasses can override this method if some concepts imply other ones.
        This enables the LIGA augmentation procedure to produce consistent sets of concept instances.
        """
        return []

    def concept_ids_to_counts(self, concept_ids: Iterable[int]) -> [int]:
        counts = [0] * len(self.concepts)
        for concept_id in concept_ids:
            counts[concept_id] += 1
            for impl_concept_id in self.get_implied_concepts(concept_id):
                counts[impl_concept_id] += 1
        return counts

    @abc.abstractmethod
    def __call__(self, image: Optional[np.ndarray], image_id: Optional[str], **kwargs) \
            -> Iterable[Tuple[int, np.ndarray]]:
        """
        Returns a list of pairs, each consisting of an interpretable concept
        and the occurrence of this concept on *image*.
        An occurrence is encoded as a boolean array that is 1 only for the pixels where the concept is located.
        *image* should be an H x W x C encoded image with values in [0., 255.].
        Alternatively, an interpreter can also require an *image_id* that identifies an image within some dataset.
        """

    def count_concepts(self, image: Optional[np.ndarray], image_id: Optional[str]) -> List[int]:
        ids_and_masks = list(self(image, image_id))
        if len(ids_and_masks) > 0:
            concept_ids, _ = list(zip(*ids_and_masks))
            return self.concept_ids_to_counts(concept_ids)
        else:
            return [0] * len(self.concepts)

    @staticmethod
    def get_overlapping_concepts(mask: np.ndarray,
                                 all_ids_and_masks: Iterable[Tuple[int, np.ndarray]],
                                 is_overlap_at: float) -> Iterable[Tuple[int, np.ndarray]]:
        overlap_ids = []
        overlap_masks = []
        overlap_masks_union = mask

        while True:
            for concept_id, other_mask in all_ids_and_masks:
                if concept_id in overlap_ids:
                    continue

                common_area = np.count_nonzero(np.bitwise_and(other_mask, overlap_masks_union))
                other_mask_area = np.count_nonzero(other_mask)
                if float(common_area) / float(other_mask_area) > is_overlap_at:
                    overlap_ids.append(concept_id)
                    overlap_masks.append(other_mask)

            new_dropped_masks_union = reduce(np.bitwise_or, overlap_masks)
            if np.all(new_dropped_masks_union == overlap_masks_union):
                break
            overlap_masks_union = new_dropped_masks_union

        return zip(overlap_ids, overlap_masks)

    def get_counterfactuals(self,
                            max_perturbed_area: float,
                            max_concept_overlap: float,
                            image: Optional[np.ndarray] = None,
                            image_id: Optional[str] = None,
                            shuffle: bool = True) -> Iterable[Tuple[List[int], np.ndarray]]:
        ids_and_masks = list(self(image=image, image_id=image_id))
        if len(ids_and_masks) > 0:
            if shuffle:
                ids_and_masks = np.asarray(ids_and_masks, dtype=object)
                self.rng.shuffle(ids_and_masks, 0)

            ids, masks = list(zip(*ids_and_masks))
            counts = self.concept_ids_to_counts(ids)
            image_area = image.shape[0] * image.shape[1]

            for dropped_id, dropped_mask in ids_and_masks:
                # the following zip cannot be empty, because dropped_mask is at least overlapping with itself
                overlapping = self.get_overlapping_concepts(mask=dropped_mask,
                                                            all_ids_and_masks=ids_and_masks,
                                                            is_overlap_at=max_concept_overlap)
                dropped_ids, dropped_masks = list(zip(*overlapping))
                dropped_area = np.count_nonzero(reduce(np.bitwise_or, dropped_masks))
                if float(dropped_area) / float(image_area) > max_perturbed_area:
                    continue

                perturbed_image = image.copy()
                dropped_counts = self.concept_ids_to_counts(dropped_ids)
                perturbed_counts = np.asarray(counts) - np.asarray(dropped_counts)
                assert np.all(perturbed_counts >= 0)
                for mask in dropped_masks:
                    # replace concept from image with grey area
                    perturbed_image[mask] = (127., 127., 127.)

                yield perturbed_counts, perturbed_image


class JoinedInterpreter(Interpreter):

    def __init__(self,
                 *interpreters: Interpreter,
                 prefix_interpreter_name: bool = True,
                 clean_concept_names: bool = True):
        super().__init__()
        self.interpreters = interpreters
        self.prefix = prefix_interpreter_name
        self.clean = clean_concept_names

        self.interpreter_id_by_concept_id = []
        for interpreter_id, interpreter in enumerate(self.interpreters):
            self.interpreter_id_by_concept_id += [interpreter_id] * len(interpreter.concepts)

    def __str__(self) -> str:
        return 'Join({})'.format(', '.join([str(i) for i in self.interpreters]))

    @property
    def concepts(self) -> [str]:
        if self.prefix:
            concepts = [str(i) + '.' + c for i in self.interpreters for c in i.concepts]
        else:
            concepts = [c for i in self.interpreters for c in i.concepts]
        if self.clean:
            concepts = [self._clean(c) for c in concepts]
        return concepts

    @staticmethod
    def _clean(concept_name):
        for c in ' ,;{}()\n\t=':
            if c in concept_name:
                concept_name = concept_name.replace(c, '_')
        return concept_name

    def get_implied_concepts(self, concept_id: int) -> Iterable[int]:
        interpreter = self.interpreters[self.interpreter_id_by_concept_id[concept_id]]
        return interpreter.get_implied_concepts(concept_id)

    def __call__(self,
                 image: Optional[np.ndarray],
                 image_id: Optional[str],
                 **kwargs) -> Iterable[Tuple[int, np.ndarray]]:
        offset = 0
        for interpreter in self.interpreters:
            for concept_id, concept_mask in interpreter(image, image_id, **kwargs):
                yield concept_id + offset, concept_mask
            offset += len(interpreter.concepts)
