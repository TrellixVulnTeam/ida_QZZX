import abc
from dataclasses import dataclass

from simexp.describe.common import DictBasedImageDescriber
from simexp.spark import Field


@dataclass
class TFDescriber(DictBasedImageDescriber, abc.ABC):
    """
    Reads batches of image data from a petastorm store
    and converts these images to Tensorflow tensors.
    Subclasses can describe these tensors as other data.

    IMPORTANT NOTE: this describer currently only works if all images have the same size.
    """
    batch_size: int

    def batch_iter(self):
        dataset = self.read_cfg.make_tf_dataset([Field.IMAGE.name, Field.IMAGE_ID.name]).batch(self.batch_size)
        for schema_view in dataset:
            ids = schema_view.image_id.numpy().astype(Field.IMAGE_ID.numpy_dtype).tolist()
            yield ids, schema_view.image
