from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from ida.common import ImageObjectProvider, RowDict
from simexp.describe.common import DictBasedImageDescriber
from simexp.spark import Field


@dataclass
class GroundTruthObjectsDescriber(DictBasedImageDescriber):
    """
    Describes each image with a set of masks for ground-truth objects,
    taken from an `ImageObjectProvider`.
    """

    # provider of object bounding boxes for given image ids
    gt_object_provider: ImageObjectProvider

    # optional subset of the dataset for which the image ids are unique
    subset: Optional[str] = None

    # whether to ignore or fail when an image has no assigned object bounding boxes
    ignore_images_without_objects: bool = True

    name: str = field(default='ground_truth_objects', init=False)

    def generate(self) -> Iterator[RowDict]:
        with self.read_cfg.make_reader(None) as reader:
            for row in reader:
                height, width = row.image.shape[:2]  # image has shape H, W, C

                try:
                    boxes = list(self.gt_object_provider.get_object_bounding_boxes(row.image_id, self.subset))
                except KeyError as e:
                    if self.ignore_images_without_objects:
                        continue
                    raise e

                masks = np.zeros((len(boxes), height, width), dtype=np.bool_)
                concept_names = []
                for mask_no, (object_name, x_0, y_0, x_1, y_1) in enumerate(boxes):
                    y_0 = np.clip(y_0 * height, 0, height).astype(int)
                    y_1 = np.ceil(np.clip(y_1 * height, 0, height)).astype(int)
                    x_0 = np.clip(x_0 * width, 0, width).astype(int)
                    x_1 = np.ceil(np.clip(x_1 * width, 0, width)).astype(int)
                    masks[mask_no, y_0:y_1, x_0:x_1] = True
                    concept_names.append(object_name)

                concept_names = np.asarray(concept_names, dtype=np.unicode_)

                yield {Field.IMAGE_ID.name: row.image_id,
                       Field.DESCRIBER.name: self.name,
                       Field.CONCEPT_NAMES.name: concept_names,
                       Field.CONCEPT_MASKS.name: masks}
