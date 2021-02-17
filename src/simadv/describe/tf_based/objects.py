import multiprocessing
import os
import queue
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from petastorm.unischema import Unischema
from simple_parsing import ArgumentParser

from simadv.common import LoggingConfig
from simadv.describe.tf_based.base import TFDescriber
from simadv.spark import Field, Schema, PetastormWriteConfig
from simadv.oiv4.metadata import OIV4MetadataProvider

mp = multiprocessing.get_context('spawn')


@dataclass
class TFObjectDetectionProcess(mp.Process):
    """
    Executes an object detection model in a separate process
    and delivers the results with a queue.
    """

    model_dir: str
    in_queue: mp.JoinableQueue
    out_queue: mp.Queue
    gpu_id: Optional[int]

    def __post_init__(self):
        super().__init__()

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

    def run(self):
        if self.gpu_id is not None:
            # set GPU id before importing tensorflow
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(self.gpu_id)

        # import tensorflow locally in the process
        import tensorflow as tf

        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            assert len(physical_devices) == 1
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        model = tf.saved_model.load(self.model_dir)
        model = model.signatures['serving_default']

        self.out_queue.put(None)  # signal readiness

        while True:
            ids, in_batch = self.in_queue.get()
            if in_batch is None:
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
    meta: OIV4MetadataProvider = field(default_factory=OIV4MetadataProvider)
    debug: bool = False
    name: str = field(default='oiv4_objects', init=False)

    @staticmethod
    def _tf_gpu_query():
        import tensorflow as tf
        return tf.config.list_physical_devices('GPU')

    @staticmethod
    def _get_gpus():
        with mp.Pool(1) as p:  # run in separate process to make sure no memory is kept allocated
            return p.apply(OIV4ObjectsImageDescriber._tf_gpu_query)

    def __post_init__(self):
        assert 0. <= self.threshold < 1.

        num_gpus = len(self._get_gpus())
        gpus = range(num_gpus) if num_gpus > 0 else (None,)

        self.in_queue = mp.JoinableQueue(maxsize=max(1, num_gpus))
        self.out_queue = mp.Queue(maxsize=self.batch_size)
        model_dir = self.meta.cache_pretrained_model(self.model_name)
        self.processes = [TFObjectDetectionProcess(model_dir, self.in_queue, self.out_queue, gpu) for gpu in gpus]

    def result_iter(self):
        for p in self.processes:
            p.start()

        # wait until processes have allocated their ML models on the GPU
        for _ in self.processes:
            start_signal = self.out_queue.get()
            assert start_signal is None

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

            if not np.any(scores_batches > self.threshold):
                continue

            height, width = image_tensors.shape[1:3]  # shape is B, H, W, C

            for image_id, image_tensor, obj_ids, scores, boxes in zip(ids, image_tensors, obj_ids_batches,
                                                                      scores_batches, boxes_batches):
                if not np.any(scores > self.threshold):
                    continue

                indices = scores > self.threshold
                concept_names = np.asarray([self.meta.object_names[obj_id] for obj_id in obj_ids[indices]],
                                           dtype=np.unicode_)
                masks = np.zeros((len(concept_names), height, width), dtype=np.bool_)

                if self.debug:
                    image = Image.fromarray(image_tensor.numpy(), 'RGB')
                    draw = ImageDraw.Draw(image)

                for mask_no, ((y_0, x_0, y_1, x_1), score) in enumerate(zip(boxes[indices], scores[indices])):
                    y_0 = np.clip(y_0 * height, 0, height).astype(int)
                    y_1 = np.ceil(np.clip(y_1 * height, 0, height)).astype(int)
                    x_0 = np.clip(x_0 * width, 0, width).astype(int)
                    x_1 = np.ceil(np.clip(x_1 * width, 0, width)).astype(int)
                    masks[mask_no, y_0:y_1, x_0:x_1] = True

                    if self.debug:
                        draw.rectangle((x_0, y_1, x_1, y_0), outline=(255, 0, 0))
                        draw.text((x_0, y_0), '{} ({})'.format(concept_names[mask_no], score), fill=(255, 0, 0))

                if self.debug:
                    image.show()
                    input('--- press enter to continue ---')

                yield {Field.IMAGE_ID.name: image_id,
                       Field.DESCRIBER.name: self.model_name,
                       Field.CONCEPT_NAMES.name: concept_names,
                       Field.CONCEPT_MASKS.name: masks}


@dataclass
class ConceptsWriteConfig(PetastormWriteConfig):
    output_schema: Unischema = field(default=Schema.CONCEPT_MASKS, init=False)


if __name__ == '__main__':
    parser = ArgumentParser(description='Describe images with visible OIV4 objects.')
    parser.add_arguments(OIV4ObjectsImageDescriber, dest='describer')
    parser.add_arguments(ConceptsWriteConfig, dest='write_cfg')
    parser.add_arguments(LoggingConfig, dest='logging')
    args = parser.parse_args()

    df = args.describer.to_df()
    args.describer.spark_cfg.write_petastorm(df, args.write_cfg)
