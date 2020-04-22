import csv
import re
from typing import Mapping, Optional
import numpy as np

import torch
from petastorm.codecs import ScalarCodec
from petastorm.unischema import UnischemaField
from pyspark.sql.types import StringType, IntegerType

from thesis_impl.hub import SupervisedImageDataset, SupervisedImageDatasetMeta
from thesis_impl.util.functools import cached_property
from thesis_impl.places365 import wideresnet
from thesis_impl.util.webcache import WebCache


class Places365HubMeta(SupervisedImageDatasetMeta):

    _dataset_name = 'Places365Challenge'
    # the longest image ID in Places365 has 42 characters
    _image_id_field = UnischemaField('image_id', np.unicode_, (),
                                     ScalarCodec(StringType()), False)
    _label_field = UnischemaField('label_id', np.int16, (),
                                  ScalarCodec(IntegerType()), False)

    @property
    def dataset_name(cls) -> str:
        return cls._dataset_name

    @property
    def image_id_field(cls):
        return cls._image_id_field

    @property
    def label_field(cls):
        return cls._label_field


class Places365Hub(SupervisedImageDataset, metaclass=Places365HubMeta):
    """
    Provides simple access to the Places365 metadata
    and some pre-trained models.

    Downloads the necessary data from the project websites on-the fly
    and caches it for future requests.
    """

    _CSAIL_DOWNLOAD_URL = 'http://data.csail.mit.edu/places/places365/'

    _CSAIL_MODELS_URL = 'http://places2.csail.mit.edu/models_places365/'

    _CSAIL_DEMO_URL = 'http://places.csail.mit.edu/demo/'

    _GITHUB_DOWNLOAD_URL = 'https://github.com/CSAILVision/places365/raw/' \
                           '6f4647534a09374acf1f5d702566071b96c9a1b8/'

    _LABEL_RE = re.compile(r'/[a-z]/(.*)')

    def __init__(self, cache: Optional[WebCache]=None):
        """
        :param cache: cache to use for downloading files
        """
        if cache is None:
            cache = WebCache('~/.cache/places-365')
        self.cache = cache

    def get_image_id(self, image_path, subset=None):
        if subset is None:
            raise ValueError('Places365 image IDs are only unique within the '
                             'train, test or validation subset respectively.'
                             'Please specify the subset.')

        if subset == 'validation':
            return image_path.name
        elif subset in ['train', 'test']:
            two_parents = image_path.parents[2]
            rel = str(image_path.relative_to(two_parents))
            return '/' + str(image_path.relative_to(two_parents))
        else:
            raise ValueError('Subset {} does not exist.'.format(subset))

    def get_image_label(self, image_path, subset=None):
        if subset is None:
            raise ValueError('To find out the correct label we must know '
                             'whether the image is part of the validation or'
                             'train subset.')

        image_id = self.get_image_id(image_path, subset)

        if subset == 'validation':
            return self.validation_label_map[image_id]
        elif subset == 'train':
            return self.train_label_map[image_id]
        else:
            raise ValueError('Only the \'train\' and \'validation\' subsets '
                             'have labels in the Places365 challenge.')

    def _prettify_label(self, label: str):
        label = self._LABEL_RE.fullmatch(label).group(1)
        label = label.replace('_', ' ').replace('/', ' - ')
        return label

    def _cache_challenge_metadata(self):
        self.cache.cache('filelist_places365-challenge.tar',
                         self._CSAIL_DOWNLOAD_URL,
                         is_archive=True)

    @cached_property
    def label_names(self) -> [str]:
        """
        Returns a list of all labels that can be predicted
        in the Places365 task.
        The index of a label in this list is the label id that is used
        in the pre-trained CNNs.
        """
        self._cache_challenge_metadata()

        with self.cache.open('categories_places365.txt') as all_labels_file:
            csv_reader = csv.reader(all_labels_file, delimiter=' ')
            return [self._prettify_label(row[0]) for row in csv_reader]

    def label_name(self, label_id):
        return self.label_names[label_id]

    @cached_property
    def indoor_outdoor_map(self) -> [bool]:
        """
        Returns a list of boolean values `l`.
        If `l[i]` is `True`, then the label with id `i`
        denotes an outdoor scene.
        If `l[i]` is `False`, the label with id `i` denotes an indoor scene.
        """
        with self.cache.open('IO_places365.txt',
                             self._GITHUB_DOWNLOAD_URL) as labels_io_file:
            csv_reader = csv.reader(labels_io_file, delimiter=' ')
            return [bool(int(row[1]) - 1) for row in csv_reader]

    def is_outdoor(self, label_id) -> bool:
        """
        Returns `True` iff the label with `label_id` denotes an outdoor scene.
        """
        return self.indoor_outdoor_map[label_id]

    @cached_property
    def validation_label_map(self) -> Mapping[str, int]:
        """
        Returns a mapping from an image file name of the validation set
        to the id of the correct label of this image.
        """
        self._cache_challenge_metadata()

        with self.cache.open('places365_val.txt') as label_map_file:
            csv_reader = csv.reader(label_map_file, delimiter=' ')
            return {row[0]: int(row[1]) for row in csv_reader}

    @cached_property
    def train_label_map(self) -> Mapping[str, int]:
        """
        Returns a mapping from an image file name of the training set
        to the id of the correct label of this image.
        """
        self._cache_challenge_metadata()

        with self.cache.open('places365_train_challenge.txt') as label_map_file:
            csv_reader = csv.reader(label_map_file, delimiter=' ')
            return {row[0]: int(row[1]) for row in csv_reader}

    def open_demo_image(self):
        return self.cache.open('6.jpg', self._CSAIL_DEMO_URL, mode='rb')

    def _convert_and_cache_resnet18(self):
        legacy_file_name = 'wideresnet18_places365.pth.tar'
        new_path = self.cache.cache_dir / 'wideresnet18_places365_converted'

        if not new_path.exists():
            with self.cache.open(legacy_file_name,
                                 self._CSAIL_MODELS_URL, 'rb')\
                    as legacy_tar_file:
                model = torch.load(legacy_tar_file,
                                   map_location=lambda storage, loc: storage,
                                   encoding='latin1')
                torch.save(model, new_path)
                (self.cache.cache_dir / legacy_file_name).unlink()

    def resnet18(self):
        """
        Returns a pre-trained ResNet18 provided by the original authors
        to predict one out of the 365 scene labels for new images.
        """
        self._convert_and_cache_resnet18()

        with self.cache.open('wideresnet18_places365_converted', mode='rb')\
                as model_file:
            model = wideresnet.resnet18(num_classes=365,
                                        normalize_channels=((.485, .456, .406),
                                                            (.229, .224, .225)))
            checkpoint = torch.load(model_file,
                                    map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v
                          for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            model.eval()
            return model

    def vote_indoor_outdoor(self, label_ids):
        """
        Returns `True` iff more labels in `label_ids` denote indoor scenes
        than outdoor scenes.
        """
        indoor_count = sum(1 for l in label_ids if self.indoor_outdoor_map[l])
        total_count = sum(1 for _ in label_ids)
        return indoor_count > total_count / 2

    def label_name_distribution(self, label_probs, label_ids):
        """
        Takes in a sequence of label probabilities and a sequence of
        corresponding label ids.
        Returns a dictionary `{l: p, ...}` mapping readable label names
        to probabilities.
        """
        labels = (self.label_names[i] for i in label_ids)
        return {l: p for l, p in zip(labels, label_probs)}
