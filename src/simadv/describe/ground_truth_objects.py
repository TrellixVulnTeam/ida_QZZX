from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
from petastorm.unischema import Unischema

from simadv.common import ImageObjectProvider, RowDict
from simadv.io import Field, Schema
from simadv.describe.common import DictBasedImageDescriber, ImageReadConfig


@dataclass
class GroundTruthObjectsDescriber(DictBasedImageDescriber):
    """
    Describes each image with a set of masks for ground-truth objects,
    taken from an `ImageObjectProvider`.
    """

    read_cfg: ImageReadConfig
    output_schema: Unischema = field(default=Schema.CONCEPT_MASKS, init=False)

    # provider of object bounding boxes for given image ids
    gt_object_provider: ImageObjectProvider

    # optional subset of the dataset for which the image ids are unique
    subset: Optional[str]

    concept_group_name: str = field(default='perceivable_colors', init=False)

    def generate(self) -> Iterator[RowDict]:
        with self.read_cfg.make_reader(None) as reader:
            for row in reader:
                height, width = row.image.shape[:2]  # image has shape H, W, C

                boxes = list(self.gt_object_provider.get_object_bounding_boxes(row.image_id, self.subset))

                masks = np.zeros((len(boxes), height, width))
                concept_names = []
                for mask_no, object_name, x_0, y_0, x_1, y_1 in enumerate(boxes):
                    masks[mask_no, y_0:y_1, x_0:x_1] = True
                    concept_names.append(object_name)

                concept_names = np.asarray(concept_names, dtype=np.unicode_)

                yield {Field.IMAGE_ID.name: row.image_id,
                       Field.CONCEPT_GROUP.name: self.concept_group_name,
                       Field.CONCEPT_NAMES.name: concept_names,
                       Field.CONCEPT_MASKS.name: masks}
