import abc
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from PIL import ImageDraw
from petastorm.unischema import Unischema
from torchvision.transforms.functional import to_pil_image

from simadv.common import RowDict
from simadv.io import Field, Schema, PetastormWriteConfig
from simadv.torch import TorchConfig
from simadv.describe.torch.base import TorchImageDescriber


@dataclass
class ObjectsWriteConfig(PetastormWriteConfig):
    output_schema: Unischema = field(default=Schema.CONCEPT_MASKS, init=False)


class BatchedTorchImageDescriber(TorchImageDescriber, abc.ABC):
    """
    Reads batches of image data from a petastorm store
    and converts these images to Torch tensors.

    Attention: If the input images can have different sizes, you *must*
    set `read_cfg.batch_size` to 1!

    Subclasses can describe the produced tensors as other data
    batch-wise by implementing the `describe_batch` method.
    """

    torch_cfg: TorchConfig

    def batch_iter(self):
        def to_tensor(batch_columns):
            row_ids, image_arrays = batch_columns
            image_tensors = torch.as_tensor(image_arrays) \
                .to(self.torch_cfg.device, torch.float) \
                .div(255)

            return row_ids, image_tensors

        current_batch = []
        reader = self.read_cfg.make_reader(None)

        for row in reader:
            current_batch.append((row.image_id, row.image))

            if len(current_batch) < self.read_cfg.batch_size:
                continue

            yield to_tensor(list(zip(*current_batch)))
            current_batch.clear()

        if current_batch:
            yield to_tensor(list(zip(*current_batch)))

    @abc.abstractmethod
    def describe_batch(self, ids, batch) -> Iterable[RowDict]:
        pass

    def cleanup(self):
        """
        Can be overridden to clean up more stuff.
        """
        torch.cuda.empty_cache()  # release all memory that can be released

    def generate(self):
        self.classifier.torch_model.to(self.torch_cfg.device)  # move model to CUDA before tensors

        with torch.no_grad():
            for ids, batch in self.batch_iter():
                yield from self.describe_batch(ids, batch)

        self.cleanup()


@dataclass
class CocoObjectsImageDescriber(BatchedTorchImageDescriber):
    """
    Translates each image to counts of detected objects from the COCO task.
    The COCO task includes 91 objects.
    """

    _OBJECT_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter',
        'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'N/A',
        'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    concept_group_name: str = field(default='coco_objects', init=False)
    threshold: float = 0.5
    debug: bool = False

    def __post_init__(self):
        assert 0. <= self.threshold < 1.

    @property
    def object_names(self):
        return CocoObjectsImageDescriber._OBJECT_NAMES

    def describe_batch(self, ids, image_tensors):
        # transform (H, W, C) images to (C, H, W) images
        height, width = image_tensors.size[1:3]  # shape is B, H, W, C -- we want H, W
        image_tensors = image_tensors.permute(0, 3, 1, 2)

        for image_id, image_tensor, result in zip(ids, image_tensors, self.classifier.torch_model(image_tensors)):
            obj_ids, boxes, scores = result['labels'], result['boxes'], result['scores']
            indices = scores > self.threshold

            concept_names = np.asarray([self.object_names[obj_id] for obj_id in obj_ids[indices]])
            masks = np.zeros((len(concept_names), height, width))
            for mask_no, x_0, y_0, x_1, y_1 in enumerate(boxes[indices]):
                masks[mask_no, y_0:y_1, x_0:x_1] = True

            if self.debug:
                boxes = result['boxes'].tolist()

                image = to_pil_image(image_tensor, 'RGB')
                draw = ImageDraw.Draw(image)
                for obj_id, box, score in zip(obj_ids[indices], boxes[indices], scores[indices]):
                    draw.rectangle(list(box))
                    draw.text(box[:2], '{} ({})'.format(self.object_names[obj_id], score))

                image.show()

                input('--- press enter to continue ---')

            yield {Field.IMAGE_ID.name: image_id,
                   Field.CONCEPT_GROUP: self.concept_group_name,
                   Field.CONCEPT_NAMES: concept_names,
                   Field.CONCEPT_MASKS: masks}
