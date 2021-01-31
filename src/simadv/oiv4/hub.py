import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Union
import numpy as np
from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.unischema import UnischemaField
from pyspark.sql.types import StringType

from simadv.hub import SupervisedImageDataset, SupervisedImageDatasetMeta
from simadv.util.functools import cached_property
from simadv.util.webcache import WebCache


class OpenImagesV4HubMeta(SupervisedImageDatasetMeta):

    _dataset_name = 'OpenImagesV4'
    _image_id_field = UnischemaField('image_id', np.unicode_, (),
                                     ScalarCodec(StringType()), False)
    BOXES_DTYPE = [('label_id', '<i2'),
                   ('x_min', '<f4'), ('x_max', '<f4'),
                   ('y_min', '<f4'), ('y_max', '<f4'),
                   ('is_occluded', '?'),
                   ('is_truncated', '?'), ('is_group_of', '?'),
                   ('is_depiction', '?'), ('is_inside', '?')]
    _label_field = UnischemaField('boxes', np.void, (None,), NdarrayCodec(),
                                  False)

    @property
    def dataset_name(cls) -> str:
        return cls._dataset_name

    @property
    def image_id_field(cls):
        return cls._image_id_field

    @property
    def label_field(cls):
        return cls._label_field


_RE_IMAGE_ID = re.compile(r'^(.*)\.jpg$')


class OpenImagesV4Hub(SupervisedImageDataset, metaclass=OpenImagesV4HubMeta):
    """
    Provides simple access to the OpenImages V4 metadata.

    Downloads the necessary data from the project websites on-the fly
    and caches it for future requests.
    """

    _DOWNLOAD_URL = 'https://storage.googleapis.com/openimages/2018_04/'

    def __init__(self, cache: Optional[WebCache] = None):
        """
        :param cache: cache to use for downloading files
        """
        if cache is None:
            cache = WebCache('~/.cache/open-images-v4')
        self.cache = cache

    def get_image_id(self, image_path: Union[str, Path], subset: Optional[str] = None):
        if isinstance(image_path, Path):
            image_path = image_path.name
        return _RE_IMAGE_ID.fullmatch(image_path).group(1)

    def get_image_label(self, image_path: Union[str, Path], subset: Optional[str] = None):
        if subset is None:
            raise ValueError('To find out the correct label we must know '
                             'whether the image is part of the train, test or'
                             'validation subset.')

        image_id = self.get_image_id(image_path, subset)
        return self.boxes_map(subset)[image_id]

    @cached_property
    def label_names(self) -> [str]:
        """
        Returns a list of all object names that have bounding boxes in the dataset.
        The index of each object name in the list is its id in output arrays of object detection models.
        """
        with self.cache.open('class-descriptions-boxable.csv', self._DOWNLOAD_URL) \
                as object_names_file:
            csv_reader = csv.reader(object_names_file, delimiter=',')
            return ['__background__'] + [row[1] for row in csv_reader]

    @cached_property
    def _label_names_map(self) -> Mapping[str, str]:
        """
        Returns a mapping from "label ids of objects" to human readable label names.
        See `label_name()` below.
        """
        with self.cache.open('class-descriptions-boxable.csv',
                             self._DOWNLOAD_URL)\
                as all_labels_file:
            csv_reader = csv.reader(all_labels_file, delimiter=',')
            return {label_id: label_name for label_id, label_name in csv_reader}

    def label_name(self, label_id: str):
        """
        Looks up the human readable label name of a "label id".
        "Label ids" in the OpenImages dataset are a special concept of the OpenImages dataset,
        they are uniform identifiers for both objects and per-image labels.
        This method only works for label ids of objects.
        """
        return self._label_names_map[label_id]

    @cached_property
    def object_names(self):
        """
        Returns a list of all object names that have bounding boxes in the dataset.
        The index of each object name in the list is its id during classification.
        """
        with self.cache.open('class-descriptions-boxable.csv', self._DOWNLOAD_URL) \
                as object_names_file:
            csv_reader = csv.reader(object_names_file, delimiter=',')
            return ['__background__'] + [row[1] for row in csv_reader]

    @lru_cache(None)
    def boxes_map(self, subset: str) -> Mapping[str, np.ndarray]:
        assert subset in ['train', 'test', 'validation']
        label_id_by_name = {label_name: label_id for label_id, label_name in enumerate(self.label_names)}

        with self.cache.open('{}-annotations-bbox.csv'.format(subset),
                             self._DOWNLOAD_URL + '/{}'.format(subset)) \
                as validation_labels_file:
            csv_reader = csv.DictReader(validation_labels_file, delimiter=',')
            boxes_row_map = {}

            for row in csv_reader:
                image_id = row['ImageID']
                del row['ImageID']

                if image_id in boxes_row_map:
                    boxes_row_map[image_id].append(row)
                else:
                    boxes_row_map[image_id] = [row]

            boxes_map = {
                image_id: np.array([(label_id_by_name[self.label_name(row['LabelName'])],
                                     float(row['XMin']), float(row['XMax']),
                                     float(row['YMin']), float(row['YMax']),
                                     bool(row['IsOccluded']),
                                     bool(row['IsTruncated']),
                                     bool(row['IsGroupOf']),
                                     bool(row['IsDepiction']),
                                     bool(row['IsInside']))
                                    for row in boxes_rows],
                                   dtype=type(self).BOXES_DTYPE)
                for image_id, boxes_rows in boxes_row_map.items()
            }

            return boxes_map

    @lru_cache(None)
    def image_label_map(self, subset: str) -> Mapping[str, np.ndarray]:
        assert subset in ['train', 'test', 'validation']
        with self.cache.open('{}-annotations-human-imagelabels-boxable.csv'.format(subset),
                             self._DOWNLOAD_URL + '/{}'.format(subset)) \
                as validation_labels_file:
            csv_reader = csv.DictReader(validation_labels_file, delimiter=',')
            return {row['ImageID']: row['LabelName'] for row in csv_reader}
