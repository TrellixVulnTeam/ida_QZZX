import json
import logging
from importlib import resources
from pathlib import Path
from typing import Optional, Tuple
from unittest import mock

import numpy as np
import torch
import torchvision
from skimage import img_as_float
from skimage.transform import resize
from torch.nn import DataParallel
from torch.nn.functional import softmax

from ida.common import Classifier
from ida.util.webcache import WebCache
from ida.torch_extensions.adjusted_resnet_basic_block import AdjustedBasicBlock


_COLLECTION_PACKAGE = 'ida.classifier_collection.torch'


class TorchImageClassifier(Classifier):

    _HASH_ATTRS = ['name', 'num_classes', 'param_url']

    def __init__(self,
                 name: str,
                 num_classes: int,
                 param_url: Optional[str],
                 is_parallel: bool,
                 memoize: bool = False,
                 use_cuda: bool = True,
                 input_size: Tuple[int, int] = (224, 224),
                 cache: WebCache = WebCache(),
                 is_latin1: bool = False,
                 json_path: Optional[str] = None):
        """

        :param name: name of this model
        :param num_classes: how many classes this classifier discriminates
        :param param_url: an optional file to load this model from.
            if None, the model will be loaded from the torchvision collection.
        :param is_parallel: whether this model uses `DataParallel`
        :param memoize: whether to cache predictions
        :param input_size: required input size of the model
        :param cache: WebCache = WebCache()
        :param is_latin1: whether this model was encoded with latin1 --
            a frequent "bug" with models exported from older torch versions
        """
        super().__init__(name=name,
                         num_classes=num_classes,
                         memoize_predictions=memoize)

        self.param_url = param_url
        self.is_parallel = is_parallel
        self.use_cuda = use_cuda
        self.input_size = input_size
        self.cache = cache
        self.is_latin1 = is_latin1
        self.json_path = json_path

        if self.param_url:
            self.url, self.file_name = self.param_url.rsplit('/', 1)
            self.url += '/'

        self._model = None

        # magic constants in the torch community
        means, sds = np.asarray([0.485, 0.456, 0.406]), np.asarray([0.229, 0.224, 0.225])
        self._means_nested = means[None, None, :]
        self._sds_nested = sds[None, None, :]

    @property
    def device(self) -> torch.device:
        return torch.device('cuda') if self.use_cuda and torch.cuda.is_available() else torch.device('cpu')

    @staticmethod
    def from_json_file(path_str: str, **kwargs):
        with resources.path(_COLLECTION_PACKAGE, path_str) as path:
            with path.open('r') as f:
                json_dict = json.load(f)
                return TorchImageClassifier(name=json_dict['name'],
                                            param_url=json_dict['param_url'],
                                            is_parallel=json_dict['is_parallel'],
                                            num_classes=json_dict['num_classes'],
                                            is_latin1=json_dict['is_latin1'],
                                            json_path=path_str,
                                            **kwargs)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_model'] = None
        return d

    def get_fingerprint(self):
        return [getattr(self, attr) for attr in self._HASH_ATTRS]

    def __hash__(self):
        return hash(self.get_fingerprint())

    def __eq__(self, other):
        if not isinstance(other, TorchImageClassifier):
            return False
        return self.get_fingerprint() == [getattr(other, attr) for attr in self._HASH_ATTRS]

    def _convert_latin1_to_unicode_and_cache(self):
        dest_path = self.cache.cache_dir / self.file_name
        legacy_path = Path(self.cache.cache_dir / (dest_path.name + '.legacy'))

        if not dest_path.exists():
            logging.info('Converting legacy latin1 encoded model to unicode...', )
            with self.cache.open(self.file_name, self.url, legacy_path.name, 'rb') as legacy_tar_file:
                model = torch.load(legacy_tar_file,
                                   map_location=self.device,
                                   encoding='latin1')
            torch.save(model, dest_path)
            legacy_path.unlink()
            logging.info('Conversion done.')

    @property
    def torch_model(self):
        with mock.patch.object(torchvision.models.resnet, 'BasicBlock', AdjustedBasicBlock):
            if self._model is not None:
                return self._model

            if self.param_url is None:
                self._model = torchvision.models.__dict__[self.name](pretrained=True)
            else:
                if self.is_latin1:
                    self._convert_latin1_to_unicode_and_cache()

                with self.cache.open(self.file_name, self.url, None, 'rb') as param_file:
                    checkpoint = torch.load(param_file, map_location=self.device)

                if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
                    self._model = torchvision.models.__dict__[self.name](num_classes=self.num_classes)
                    if self.is_parallel:
                        # the data parallel layer will add 'module' before each layer name:
                        state_dict = {str.replace(k, 'module.', ''): v
                                      for k, v in checkpoint['state_dict'].items()}
                    else:
                        state_dict = checkpoint
                    self._model.load_state_dict(state_dict)
                else:
                    self._model = checkpoint

        self._model.to(self.device)
        self._model.eval()
        if self.is_parallel and torch.cuda.device_count() > 1:
            self._model = DataParallel(self._model)
        return self._model

    @property
    def torch_module(self):
        if isinstance(self.torch_model, DataParallel):
            return self.torch_model.module
        return self.torch_model

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

        self.torch_model.to(self.device)
        image_tensors = torch.from_numpy(inputs).float().to(self.device)
        logits = self.torch_model(image_tensors)
        probs = softmax(logits, dim=1)
        return probs.detach().cpu().numpy()
