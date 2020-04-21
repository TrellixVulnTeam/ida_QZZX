import csv
import re
from typing import Mapping, Optional
import numpy as np
from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.unischema import UnischemaField
from pyspark.sql.types import StringType

from thesis_impl.hub import SupervisedImageDataset, SupervisedImageDatasetMeta
from thesis_impl.util.functools import cached_property
from thesis_impl.util.webcache import WebCache


class OpenImagesV4HubMeta(SupervisedImageDatasetMeta):

    _dataset_name = 'OpenImagesV4'
    _image_id_field = UnischemaField('image_id', np.unicode_, (),
                                     ScalarCodec(StringType()), False)
    BOXES_DTYPE = [('label_id', '<U16'),
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
    _VALIDATION_URL = _DOWNLOAD_URL + 'validation/'

    def __init__(self, cache: Optional[WebCache]=None):
        """
        :param cache: cache to use for downloading files
        """
        if cache is None:
            cache = WebCache('~/.cache/open-images-v4')
        self.cache = cache

    def get_image_id(self, image_path, subset=None):
        return _RE_IMAGE_ID.fullmatch(image_path.name).group(1)

    def get_image_label(self, image_path, subset=None):
        if subset is None:
            raise ValueError('To find out the correct label we must know '
                             'whether the image is part of the train, test or'
                             'validation subset.')

        image_id = self.get_image_id(image_path, subset)

        if subset == 'validation':
            return self.validation_boxes_map[image_id]
        else:
            raise NotImplementedError()

    @cached_property
    def _label_names(self) -> Mapping[str, str]:
        """
        Returns a mapping from label IDs to human readable label names.
        """
        with self.cache.open('class-descriptions-boxable.csv',
                             self._DOWNLOAD_URL)\
                as all_labels_file:
            csv_reader = csv.reader(all_labels_file, delimiter=',')
            return {label_id: label_name for label_id, label_name in csv_reader}

    def label_name(self, label_id: str):
        return self._label_names[label_id]

    @cached_property
    def validation_boxes_map(self) -> Mapping[str, np.ndarray]:
        """
        Returns a mapping from an image ID of the validation set
        to the list of boxes and labels that were annotated for this image.
        """
        labels_file_name = 'validation-annotations-bbox.csv'

        with self.cache.open(labels_file_name, self._VALIDATION_URL) \
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
                image_id: np.array([(row['LabelName'],
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

    @cached_property
    def validation_image_label_map(self) -> Mapping[str, str]:
        """
        Returns a mapping from an image ID of the validation set
        to the id of the correct image level label of this image.
        """
        labels_file_name = 'validation-annotations-human-imagelabels.csv'

        with self.cache.open(labels_file_name, self._VALIDATION_URL) \
                as validation_labels_file:
            csv_reader = csv.DictReader(validation_labels_file, delimiter=',')
            return {row['ImageID']: row['LabelName'] for row in csv_reader}
