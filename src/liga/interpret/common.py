import abc
from typing import Optional, Tuple, Iterable

import numpy as np


class Interpreter(abc.ABC):

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


class JoinedInterpreter(Interpreter):

    def __init__(self,
                 *interpreters: Interpreter,
                 prefix_interpreter_name: bool = True,
                 clean_concept_names: bool = True):
        super().__init__()
        self.interpreters = interpreters
        self.prefix = prefix_interpreter_name
        self.clean = clean_concept_names

    def __str__(self) -> str:
        return 'Join({})'.format(', '.join([str(i) for i in self.interpreters]))

    @property
    def concepts(self) -> [str]:
        if self.prefix:
            concepts = [str(i) + '.' + c for i in self.interpreters for c in i.concepts]
        else:
            concepts = [c for i in self.interpreters for c in i.concepts]
        if self.clean:
            concepts = list(map(self._clean, concepts))
        return concepts

    @staticmethod
    def _clean(concept_name):
        for c in ' ,;{}()\n\t=':
            if c in concept_name:
                concept_name = concept_name.replace(c, '_')
        return concept_name

    def __call__(self,
                 image: Optional[np.ndarray],
                 image_id: Optional[str],
                 **kwargs) -> Iterable[Tuple[int, np.ndarray]]:
        offset = 0
        for interpreter in self.interpreters:
            for concept_id, concept_mask in interpreter(image, image_id, **kwargs):
                yield concept_id + offset, concept_mask
            offset += len(interpreter.concepts)
