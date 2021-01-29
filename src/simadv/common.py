import abc
import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torchvision
from petastorm import make_reader
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.tf_utils import make_petastorm_dataset
from petastorm.unischema import UnischemaField
from skimage import img_as_float
from skimage.transform import resize
from torch.nn.functional import softmax

from pyspark.sql import SparkSession
from simple_parsing import Serializable

from simadv.util.webcache import WebCache


@dataclass
class LoggingConfig:
    level: str = 'info'  # the logging level

    def __post_init__(self):
        logging.basicConfig(level=str.upper(self.level))


@dataclass
class TorchConfig:
    use_cuda: bool = True  # whether to use CUDA if available
    read_batch_size: int = 64  # number of samples to read at once
    read_num_workers: int = 4  # number of parallel data loading processes

    def __post_init__(self):
        self.use_cuda = self.use_cuda and torch.cuda.is_available()
        self.device: torch.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

    def set_device(self):
        """
        Context manager that executes the contained code with the
        configured torch device.
        """
        return torch.cuda.device(self.device if self.use_cuda else -1)


@dataclass
class PetastormReadConfig:
    input_url: str
    batch_size: int
    shuffle: bool = False
    pool_type: str = 'thread'
    workers_count: int = 10

    def make_reader(self, schema_fields: [str], **kwargs):
        return make_reader(self.input_url,
                           shuffle_row_groups=self.shuffle,
                           reader_pool_type=self.pool_type,
                           workers_count=self.workers_count,
                           schema_fields=schema_fields,
                           **kwargs)

    def make_tf_dataset(self, schema_fields: [str]):
        return make_petastorm_dataset(self.make_reader(schema_fields)).batch(self.batch_size)


@dataclass
class PetastormWriteConfig:
    spark_master: str
    spark_driver_memory: str
    spark_exec_memory: str
    row_size: int

    def __post_init__(self):
        self.builder = SparkSession.builder \
            .config('spark.driver.memory',
                    self.spark_driver_memory) \
            .config('spark.executor.memory',
                    self.spark_exec_memory) \
            .master(self.spark_master)

    @property
    def session(self):
        return self.builder.getOrCreate()

    def write_parquet(self, out_df, schema, output_url):
        while True:
            try:
                with materialize_dataset(self.session, output_url,
                                         schema, self.row_size):
                    out_df.write.mode('error').parquet(output_url)
            except Exception as e:
                logging.error('Encountered exception: {}'.format(e))
                other_url = input('To retry, enter another '
                                  'output URL and press <Enter>.'
                                  'To exit, just press <Enter>.')
                if not other_url:
                    raise e
                output_url = other_url
            else:
                break


@dataclass
class PetastormTransformer:
    id_field: UnischemaField
    output_url: str
    read_cfg: PetastormReadConfig
    write_cfg: PetastormWriteConfig


@dataclass
class ClassificationTask:
    name: str
    num_classes: int


@dataclass
class Classifier(abc.ABC):
    """
    Abstract base class for classifiers.
    """

    # name of this model
    name: str

    # the task this model was trained on
    task: ClassificationTask

    @abc.abstractmethod
    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts probability distributions NxC over the C classes for the given N inputs.
        The distributions are encoded as a float array.
        """
        pass


@dataclass
class TorchImageClassifier(Classifier, Serializable):

    # an optional file to load this model from
    # if None, the model will be loaded from the torchvision collection.
    param_url: Optional[str]

    # whether this model uses `DataParallel`
    is_parallel: bool

    # torch configuration to use
    torch_cfg: TorchConfig

    # required input size of the model
    input_size: Tuple[int, int] = (224, 224)

    # where to cache downloaded information
    cache: WebCache = WebCache()

    # whether this model was encoded with latin1 (a frequent "bug" with models exported from older torch versions)
    is_latin1: bool = False

    def __post_init__(self):
        if self.param_url:
            self.url, self.file_name = self.param_url.rsplit('/', 1)
            self.url += '/'

        self._model = None

        # magic constants in the torch community
        means, sds = np.asarray([0.485, 0.456, 0.406]), np.asarray([0.229, 0.224, 0.225])
        self._means_nested = means[None, None, :]
        self._sds_nested = sds[None, None, :]

    def _convert_latin1_to_unicode_and_cache(self):
        dest_path = self.cache.cache_dir / self.file_name
        legacy_path = Path(self.cache.cache_dir / (dest_path.name + '.legacy'))

        if not dest_path.exists():
            logging.info('Converting legacy latin1 encoded model to unicode...', )
            with self.cache.open(self.file_name, self.url, legacy_path.name, 'rb') as legacy_tar_file:
                model = torch.load(legacy_tar_file,
                                   map_location=self.torch_cfg.device,
                                   encoding='latin1')
            torch.save(model, dest_path)
            legacy_path.unlink()
            logging.info('Conversion done.')

    @property
    def torch_model(self):
        if self.torch_cfg is None:
            raise RuntimeError('Attribute torch_cfg must be set before the model can be accessed.')

        if self._model is not None:
            return self._model

        if self.param_url is None:
            self._model = torchvision.models.__dict__[self.name](pretrained=True)
        else:
            if self.is_latin1:
                self._convert_latin1_to_unicode_and_cache()

            with self.cache.open(self.file_name, self.url, None, 'rb') as param_file:
                checkpoint = torch.load(param_file,
                                        map_location=self.torch_cfg.device)
            if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
                self._model = torchvision.models.__dict__[self.name](num_classes=self.task.num_classes)
                if self.is_parallel:
                    # the data parallel layer will add 'module' before each layer name:
                    state_dict = {str.replace(k, 'module.', ''): v
                                  for k, v in checkpoint['state_dict'].items()}
                else:
                    state_dict = checkpoint
                self._model.load_state_dict(state_dict)
            else:
                self._model = checkpoint

        self._model.eval()
        return self._model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Takes in an RGB `image` in shape (H, W, C) with value range [0, 255]
        and outputs a resized and normalized image in shape (C, H, W).
        """
        image = img_as_float(image)
        image = (resize(image, self.input_size) - self._means_nested) / self._sds_nested
        return np.transpose(image, (2, 0, 1))

    def predict_proba(self, inputs: np.ndarray, preprocessed=False) -> np.ndarray:
        if not preprocessed:
            inputs = np.asarray([self.preprocess(img) for img in inputs])

        self.torch_model.to(self.torch_cfg.device)
        image_tensors = torch.from_numpy(inputs).float().to(self.torch_cfg.device)
        logits = self.torch_model(image_tensors)
        probs = softmax(logits, dim=1)
        return probs.detach().cpu().numpy()


@dataclass
class TorchImageClassifierSerialization:
    path: str  # path to a json serialization of a TorchImageClassifier

    _COLLECTION_PACKAGE = 'simadv.classifier_collection'

    def _raise_invalid_path(self):
        raise ValueError('Classifier path {} is neither a valid path, nor does it point to a valid resource.'
                         .format(self.path))

    def __post_init__(self):
        if not Path(self.path).exists():
            try:
                if self.path.endswith('.json') and resources.is_resource(self._COLLECTION_PACKAGE, self.path):
                    with resources.path(self._COLLECTION_PACKAGE, self.path) as path:
                        self.path = str(path)
                else:
                    self._raise_invalid_path()
            except ModuleNotFoundError:
                self._raise_invalid_path()


RowDict = Dict[str, Any]