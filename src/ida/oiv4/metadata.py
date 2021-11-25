import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Union, Iterable, Tuple

import numpy as np

from ida.common import ImageIdProvider, ImageObjectProvider
from ida.util.functools import cached_property
from ida.util.webcache import WebCache

_RE_IMAGE_ID = re.compile(r'^(.*)\.jpg$')


@dataclass
class OIV4CacheMixin:
    MODELS_URL = 'http://download.tensorflow.org/models/object_detection/'
    LABELS_URL = 'https://storage.googleapis.com/openimages/2018_04/'

    cache: WebCache = field(default_factory=lambda: WebCache('~/.cache/open-images-v4'))

    def cache_pretrained_model(self, model_name) -> str:
        """
        Caches the model `model_name` and returns the path to its directory.
        """
        self.cache.cache(model_name + '.tar.gz', self.MODELS_URL, is_archive=True)
        return str(self.cache.get_absolute_path(model_name) / 'saved_model')


class OIV4MetadataProvider(ImageIdProvider, ImageObjectProvider, OIV4CacheMixin):
    """
    Provides image ids and object bounding boxes for the OpenImages V4 dataset.
    """

    BOXES_DTYPE = [('class_id', '<i2'),
                   ('x_min', '<f4'), ('x_max', '<f4'),
                   ('y_min', '<f4'), ('y_max', '<f4'),
                   ('is_occluded', '?'),
                   ('is_truncated', '?'), ('is_group_of', '?'),
                   ('is_depiction', '?'), ('is_inside', '?')]

    LABELS_URL = 'https://storage.googleapis.com/openimages/2018_04/'

    _boxes_cache = {}

    def get_image_id(self, image_path: Union[str, Path], subset: Optional[str] = None):
        if isinstance(image_path, Path):
            image_path = image_path.name
        return _RE_IMAGE_ID.fullmatch(image_path).group(1)

    def get_object_bounding_boxes(self, image_id: str, subset: Optional[str] = None) \
            -> Iterable[Tuple[str, float, float, float, float]]:
        if subset is None:
            if image_id in self.boxes_map('train'):
                subset = 'train'
            elif image_id in self.boxes_map('test'):
                subset = 'test'
            elif image_id in self.boxes_map('validation'):
                subset = 'validation'
            else:
                raise KeyError('No boxes found for the given image id.')

        for box in self.boxes_map(subset)[image_id]:
            yield box['class_id'], box['x_min'], box['y_min'], box['x_max'], box['y_max']

    @cached_property
    def _get_implication_map(self) -> Mapping[int, Iterable[int]]:
        """
        Returns a mapping from "label ids of objects" to human readable label names.
        See `object_name_for_label_id()` below.
        """
        def descend(impl_tree, impl_map):
            object_name = self.object_name_for_label_id(impl_tree['LabelName'])
            object_id = self.object_names.index(object_name)

            if 'Subcategory' in impl_tree:
                for child in impl_tree['Subcategory']:
                    child_object_name = self.object_name_for_label_id(child['LabelName'])
                    child_object_id = self.object_names.index(child_object_name)
                    impl_map[child_object_id].append(object_id)
                    for prev_impl_obj_id in impl_map[object_id]:
                        # compute transitive closure
                        impl_map[child_object_id].append(prev_impl_obj_id)
                    descend(child, impl_map)

        with self.cache.open('bbox_labels_600_hierarchy.json', self.LABELS_URL) \
                as implications_file:
            tree = json.load(implications_file)
            impl_map = {object_id: [] for object_id in range(len(self.object_names))}
            for child in tree['Subcategory']:  # skip
                descend(child, impl_map)
            return impl_map

    def get_implied_objects(self, object_id: int):
        return self._get_implication_map[object_id]

    @cached_property
    def _object_names_by_label_id(self) -> Mapping[str, str]:
        """
        Returns a mapping from "label ids of objects" to human readable label names.
        See `object_name_for_label_id()` below.
        """
        with self.cache.open('class-descriptions-boxable.csv', self.LABELS_URL)\
                as all_labels_file:
            csv_reader = csv.reader(all_labels_file, delimiter=',')
            return {label_id: label_name for label_id, label_name in csv_reader}

    def object_name_for_label_id(self, label_id: str):
        """
        Looks up the human readable label name of a "label id".
        "Label ids" are a special concept of the OpenImages dataset,
        they are uniform identifiers for both objects and per-image labels.
        This method only works for label ids of objects.
        """
        return self._object_names_by_label_id[label_id]

    @cached_property
    def object_names(self):
        """
        Returns a list of all object names that have bounding boxes in the dataset.
        The index of each object name in the list is its id during classification.
        """
        with self.cache.open('class-descriptions-boxable.csv', self.LABELS_URL) \
                as object_names_file:
            csv_reader = csv.reader(object_names_file, delimiter=',')
            return ['__background__'] + [row[1] for row in csv_reader]

    def boxes_map(self, subset: str) -> Mapping[str, np.ndarray]:
        try:
            return self._boxes_cache[subset]
        except KeyError:
            pass

        assert subset in ['train', 'test', 'validation']
        object_id_by_name = {label_name: object_id for object_id, label_name in enumerate(self.object_names)}

        with self.cache.open('{}-annotations-bbox.csv'.format(subset),
                             self.LABELS_URL + '{}/'.format(subset)) \
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
                image_id: np.array([(object_id_by_name[self.object_name_for_label_id(row['LabelName'])],
                                     float(row['XMin']), float(row['XMax']),
                                     float(row['YMin']), float(row['YMax']),
                                     bool(row['IsOccluded']),
                                     bool(row['IsTruncated']),
                                     bool(row['IsGroupOf']),
                                     bool(row['IsDepiction']),
                                     bool(row['IsInside']))
                                    for row in boxes_rows],
                                   dtype=self.BOXES_DTYPE)
                for image_id, boxes_rows in boxes_row_map.items()
            }

            self._boxes_cache[subset] = boxes_map
            return boxes_map
