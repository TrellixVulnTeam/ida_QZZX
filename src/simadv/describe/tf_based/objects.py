import abc
import multiprocessing
import os
import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from simadv.common import LoggingMixin
from simadv.describe.common import DictBasedImageDescriber, ImageReadConfig
from simadv.io import Field
from simadv.oiv4.metadata import OIV4MetadataProvider

mp = multiprocessing.get_context('spawn')


class TFDescriber(DictBasedImageDescriber, abc.ABC):
    """
    Reads batches of image data from a petastorm store
    and converts these images to Tensorflow tensors.
    Subclasses can describe these tensors as other data.

    FIXME: this describer currently only works if all images have the same size
    """
    read_cfg: ImageReadConfig

    def batch_iter(self):
        for schema_view in self.read_cfg.make_tf_dataset([Field.IMAGE.name, Field.IMAGE_ID.name]):
            ids = schema_view.image_id.numpy().astype(Field.IMAGE_ID.numpy_dtype).tolist()
            yield ids, schema_view.image


@dataclass
class TFObjectDetectionProcess(mp.Process, LoggingMixin):
    """
    Executes an object detection model in a separate process
    and delivers the results with a queue.
    """

    model_dir: str
    in_queue: mp.JoinableQueue
    out_queue: mp.Queue
    gpu_id: Optional[int]

    def run(self):
        if self.gpu_id is not None:
            # set GPU id before importing tensorflow
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(self.gpu_id)

        # import tensorflow locally in the process
        import tensorflow as tf

        model = tf.saved_model.load(self.model_dir)
        model = model.signatures['serving_default']

        while True:
            ids, in_batch = self.in_queue.get()
            if in_batch is None:
                self._log_item('Exiting Process for GPU {}'.format(self.gpu_id))
                self.in_queue.task_done()
                break

            out_batch = model(in_batch)
            self.out_queue.put((ids, in_batch, out_batch))
            self.in_queue.task_done()


@dataclass
class OIV4ObjectsImageDescriber(TFDescriber):
    """
    Translates each image to a set of detected objects from the OpenImages task.
    The OpenImages task includes 600 objects.
    """

    threshold: float
    model_name: str
    meta: OIV4MetadataProvider
    debug: bool = False

    def __post_init__(self):
        assert 0. <= self.threshold < 1.

        num_gpus = torch.cuda.device_count()  # sadly, tf_based does not have an equally nice method
        gpus = range(num_gpus) if num_gpus > 0 else (None,)

        self.in_queue = mp.JoinableQueue(maxsize=max(1, num_gpus))
        self.out_queue = mp.Queue(maxsize=self.read_cfg.batch_size)
        model_dir = self.meta.cache_pretrained_model(self.model_name)
        self.processes = [TFObjectDetectionProcess(model_dir, self.in_queue, self.out_queue, gpu) for gpu in gpus]

    def result_iter(self):
        for p in self.processes:
            p.start()

        pending_count = 0
        for ids, batch in self.batch_iter():
            self.in_queue.put((ids, batch))  # blocks until a slot is free
            pending_count += 1

            try:
                yield self.out_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                pending_count -= 1

        for _ in self.processes:
            self.in_queue.put((None, None))
        self.in_queue.join()  # wait until all results are there

        for _ in range(pending_count):
            yield self.out_queue.get()

        for p in self.processes:
            p.join()

    def generate(self):
        for ids, image_tensors, results in self.result_iter():
            obj_ids_batches = results['detection_classes'].numpy().astype(int)
            scores_batches = results['detection_scores'].numpy()
            boxes_batches = results['detection_boxes'].numpy()

            height, width = image_tensors.shape[1:3]  # shape is B, H, W, C

            for image_id, image_tensor, obj_ids, scores, boxes in zip(ids, image_tensors, obj_ids_batches,
                                                                      scores_batches, boxes_batches):
                indices = scores > self.threshold
                concept_names = np.asarray([self.meta.object_names[obj_id] for obj_id in obj_ids[indices]])
                masks = np.zeros((len(concept_names), height, width))

                if self.debug:
                    image = Image.fromarray(image_tensor.numpy(), 'RGB')
                    draw = ImageDraw.Draw(image)

                for mask_no, y_0, x_0, y_1, x_1, score in enumerate(zip(boxes[indices], scores[indices])):
                    y_0 = np.clip(y_0 * height, 0, height).astype(int)
                    y_1 = np.ceil(np.clip(y_1 * height, 0, height))
                    x_0 = np.clip(x_0 * width, 0, width).astype(int)
                    x_1 = np.ceil(np.clip(x_1 * width, 0, width))
                    masks[mask_no, y_0:y_1, x_0:x_1] = True

                    if self.debug:
                        draw.rectangle((x_0, y_1, x_1, y_0), outline=(255, 0, 0))
                        draw.text((x_0, y_0), '{} ({})'.format(concept_names[mask_no], score), fill=(255, 0, 0))

                if self.debug:
                    image.show()
                    input('--- press enter to continue ---')

                yield {Field.IMAGE_ID.name: image_id,
                       Field.CONCEPT_GROUP: self.concept_group_name,
                       Field.CONCEPT_NAMES: concept_names,
                       Field.CONCEPT_MASKS: masks}
