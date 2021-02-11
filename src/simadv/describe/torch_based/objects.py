from dataclasses import dataclass, field

import numpy as np
from PIL import ImageDraw
from petastorm.unischema import Unischema
from torchvision.transforms.functional import to_pil_image

from simadv.describe.torch_based.base import BatchedTorchImageDescriber
from simadv.io import Field, Schema


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

    output_schema: Unischema = field(default=Schema.CONCEPT_MASKS, init=False)
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
