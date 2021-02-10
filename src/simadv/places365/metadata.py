import csv
import re
from dataclasses import field, dataclass
from typing import Mapping

from simadv.common import ImageIdProvider, ImageClassProvider, ClassificationTask
from simadv.util.functools import cached_property
from simadv.util.webcache import WebCache


@dataclass
class Places365CacheMixin:
    CSAIL_DOWNLOAD_URL = 'http://data.csail.mit.edu/places/places365/'
    CSAIL_DEMO_URL = 'http://places.csail.mit.edu/demo/'
    GITHUB_DOWNLOAD_URL = 'https://github.com/CSAILVision/places365/raw/' \
                          '6f4647534a09374acf1f5d702566071b96c9a1b8/'

    cache: WebCache = field(default_factory=lambda: WebCache('~/.cache/places365'))

    def cache_challenge_metadata(self):
        self.cache.cache('filelist_places365-challenge.tar', self.CSAIL_DOWNLOAD_URL, is_archive=True)


@dataclass
class Places365MetadataProvider(ImageIdProvider, ImageClassProvider, Places365CacheMixin):
    """
    Provides image ids and classes for images from Places365 dataset.
    """

    name: str = field(default='Places365Challenge', init=False)
    num_classes: str = field(default=365, init=False)

    def get_image_id(self, image_path, subset=None):
        if subset is None:
            raise ValueError('Places365 image IDs are only unique within the '
                             'train, test or validation subset respectively.'
                             'Please specify the subset.')

        if subset == 'validation':
            return image_path.name
        elif subset in ['train', 'test']:
            two_parents = image_path.parents[2]
            return '/' + str(image_path.relative_to(two_parents))
        else:
            raise ValueError('Subset {} does not exist.'.format(subset))

    def get_image_class(self, image_id, subset=None):
        if subset is None:
            raise ValueError('To find out the correct label we must know '
                             'whether the image is part of the validation or'
                             'train subset.')

        if subset == 'validation':
            return self.validation_class_map[image_id]
        elif subset == 'train':
            return self.train_class_map[image_id]
        else:
            raise ValueError('Only the \'train\' and \'validation\' subsets '
                             'have labels in the Places365 challenge.')

    @cached_property
    def validation_class_map(self) -> Mapping[str, int]:
        """
        Returns a mapping from an image file name of the validation set
        to the id of the correct class of this image.
        """
        self.cache_challenge_metadata()

        with self.cache.open('places365_val.txt') as label_map_file:
            csv_reader = csv.reader(label_map_file, delimiter=' ')
            return {row[0]: int(row[1]) for row in csv_reader}

    @cached_property
    def train_class_map(self) -> Mapping[str, int]:
        """
        Returns a mapping from an image file name of the training set
        to the id of the correct class of this image.
        """
        self.cache_challenge_metadata()

        with self.cache.open('places365_train_challenge.txt') as label_map_file:
            csv_reader = csv.reader(label_map_file, delimiter=' ')
            return {row[0]: int(row[1]) for row in csv_reader}


class Places365Task(ClassificationTask, Places365CacheMixin):
    """
    The Places365 scene classification task.
    """

    _LABEL_RE = re.compile(r'/[a-z]/(.*)')

    def _prettify_label(self, label: str):
        label = self._LABEL_RE.fullmatch(label).group(1)
        label = label.replace('_', ' ').replace('/', ' - ')
        return label

    @cached_property
    def class_names(self) -> [str]:
        """
        Returns a list of all classes that can be predicted in the Places365 task.
        The index of a label in this list is the label id that is used in the pre-trained CNNs.
        """
        self.cache_challenge_metadata()

        with self.cache.open('categories_places365.txt') as all_labels_file:
            csv_reader = csv.reader(all_labels_file, delimiter=' ')
            return [self._prettify_label(row[0]) for row in csv_reader]


class Places365IOMetadataProvider(Places365MetadataProvider, Places365CacheMixin):
    """
    Provides image ids and boolean indoor/outdoor classes for images from Places365 dataset.
    """
    def get_image_class(self, image_id, subset=None) -> int:
        scene_class = super().get_image_class(image_id, subset)
        return int(self.is_outdoor(scene_class))

    @cached_property
    def indoor_outdoor_map(self) -> [bool]:
        """
        Returns a list of boolean values `l`.
        If `l[i]` is `True`, then the label with id `i` denotes an outdoor scene, otherwise an indoor scene.
        """
        with self.cache.open('IO_places365.txt', self.GITHUB_DOWNLOAD_URL) as labels_io_file:
            csv_reader = csv.reader(labels_io_file, delimiter=' ')
            return [bool(int(row[1]) - 1) for row in csv_reader]

    def is_outdoor(self, class_id) -> bool:
        """
        Returns `True` iff the label with `label_id` denotes an outdoor scene.
        """
        return self.indoor_outdoor_map[class_id]

    def open_demo_image(self):
        return self.cache.open('6.jpg', self.CSAIL_DEMO_URL, mode='rb')

    def vote_indoor_outdoor(self, class_ids):
        """
        Returns `True` iff more labels in `label_ids` denote indoor scenes
        than outdoor scenes.
        """
        indoor_count = sum(1 for c in class_ids if self.indoor_outdoor_map[c])
        total_count = sum(1 for _ in class_ids)
        return indoor_count > total_count / 2


class Places365IOTask(Places365Task):
    """
    The Places365 indoor/outdoor classification task.
    """

    @cached_property
    def class_names(self) -> [str]:
        return ['indoor', 'outdoor']
