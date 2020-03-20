import csv
import logging
import re
from pathlib import Path
from typing import Mapping
from urllib.request import urlretrieve
import tarfile

import torch

from thesis_impl.util.functools import cached_property
from thesis_impl.places365 import wideresnet


class Places365Hub:
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

    def __init__(self, cache_dir='~/.cache/places-365'):
        """
        :param cache_dir: where to cache downloaded files
        """
        self._cache_dir = Path(cache_dir).expanduser()

    def _download(self, file_name: str, download_url: str):
        target_url = download_url + file_name
        logging.info('Downloading {}...'.format(target_url))
        urlretrieve(target_url, str(self._cache_dir / file_name))

    def _cache(self, file_name: str, download_url: str,
               is_archive=False):
        """
        If not yet cached, download `file_name` from `download_url`.
        If `is_archive` is `True`, attempt to extract contents with tar.
        Returns `True` iff there was already a file called `file_name`
        in the cache.
        """
        self._cache_dir.mkdir(exist_ok=True)
        file_path = self._cache_dir / file_name
        if not file_path.exists():
            self._download(file_name, download_url)
            if is_archive:
                with tarfile.open(file_path) as archive_file:
                    archive_file.extractall(self._cache_dir)
                    file_path.unlink()
                    file_path.touch()  # leave proof that archive was extracted
            return False
        return True

    def _open(self, file_name: str, download_url: str=None, *args, **kwargs):
        if download_url is not None:
            self._cache(file_name, download_url)
        return (self._cache_dir / file_name).open(*args, **kwargs)

    def _prettify_label(self, label: str):
        label = self._LABEL_RE.fullmatch(label).group(1)
        label = label.replace('_', ' ').replace('/', ' - ')
        return label

    def _cache_challenge_metadata(self):
        self._cache('filelist_places365-challenge.tar',
                    self._CSAIL_DOWNLOAD_URL,
                    is_archive=True)

    @cached_property
    def all_labels(self) -> [str]:
        """
        Returns a list of all labels that can be predicted
        in the Places365 task.
        The index of a label in this list is the label id that is used
        in the pre-trained CNNs.
        """
        self._cache_challenge_metadata()

        with self._open('categories_places365.txt') as all_labels_file:
            csv_reader = csv.reader(all_labels_file, delimiter=' ')
            return [self._prettify_label(row[0]) for row in csv_reader]

    def label_name(self, label_id):
        return self.all_labels[label_id]

    @cached_property
    def indoor_outdoor_map(self) -> [bool]:
        """
        Returns a list of boolean values `l`.
        If `l[i]` is `True`, then the label with id `i`
        denotes an outdoor scene.
        If `l[i]` is `False`, the label with id `i` denotes an indoor scene.
        """
        with self._open('IO_places365.txt',
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

        with self._open('places365_val.txt') as label_map_file:
            csv_reader = csv.reader(label_map_file, delimiter=' ')
            return {row[0]: int(row[1]) for row in csv_reader}

    @cached_property
    def train_label_map(self) -> Mapping[str, int]:
        """
        Returns a mapping from an image file name of the training set
        to the id of the correct label of this image.
        """
        self._cache_challenge_metadata()

        with self._open('places365_train_challenge.txt') as label_map_file:
            csv_reader = csv.reader(label_map_file, delimiter=' ')
            return {row[0]: int(row[1]) for row in csv_reader}

    def open_demo_image(self):
        return self._open('6.jpg', self._CSAIL_DEMO_URL, mode='rb')

    def _convert_and_cache_resnet18(self):
        legacy_file_name = 'wideresnet18_places365.pth.tar'
        new_file_path = self._cache_dir / 'wideresnet18_places365_converted'

        if not new_file_path.exists():
            with self._open(legacy_file_name,
                            self._CSAIL_MODELS_URL, 'rb') as legacy_tar_file:
                model = torch.load(legacy_tar_file,
                                   map_location=lambda storage, loc: storage,
                                   encoding='latin1')
                torch.save(model, new_file_path)
                (self._cache_dir / legacy_file_name).unlink()

    def resnet18(self):
        """
        Returns a pre-trained ResNet18 provided by the original authors
        to predict one out of the 365 scene labels for new images.
        """
        self._convert_and_cache_resnet18()

        with self._open('wideresnet18_places365_converted',
                        mode='rb') as model_file:
            model = wideresnet.resnet18(num_classes=365)
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
        labels = (self.all_labels[i] for i in label_ids)
        return {l: p for l, p in zip(labels, label_probs)}
