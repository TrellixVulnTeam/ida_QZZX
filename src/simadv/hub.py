import abc
from pathlib import Path
from typing import Optional

from petastorm.unischema import UnischemaField


class SupervisedImageDatasetMeta(abc.ABCMeta):
    """
    This metaclass provides class-level access to certain properties.
    These properties are independent of the instance of the dataset.
    Subclasses of `SupervisedImageDataset` should use a subclass of this
    metaclass as metaclass.
    """

    @property
    @abc.abstractmethod
    def dataset_name(cls) -> str:
        """
        The name of the dataset.
        """
        pass

    @property
    @abc.abstractmethod
    def image_id_field(cls) -> UnischemaField:
        """
        A field that encodes image IDs in this dataset.
        """
        pass

    @property
    @abc.abstractmethod
    def label_field(cls) -> UnischemaField:
        """
        A field that encodes image labels in this dataset.
        """
        pass


class SupervisedImageDataset(abc.ABC, metaclass=SupervisedImageDatasetMeta):
    """
    Provides simple access to metadata for a supervised image dataset.
    """

    @property
    def dataset_name(self) -> str:
        """
        The name of the dataset.
        """
        return type(self).dataset_name

    @property
    def image_id_field(self) -> UnischemaField:
        """
        A field that encodes image IDs in this dataset.
        """
        return type(self).image_id_field

    @property
    def label_field(self) -> UnischemaField:
        """
        A field that encodes image labels in this dataset.
        """
        return type(self).label_field

    @abc.abstractmethod
    def get_image_id(self, image_path: Path, subset: Optional[str] = None):
        """
        Returns the image id of the image located at `image_path`.
        In some datasets, the image id depends on the `subset`, i.e. validation
        or train subset.
        In this case, the subset must be specified.
        """
        pass

    @abc.abstractmethod
    def get_image_label(self, image_path: Path, subset: Optional[str] = None):
        """
        Returns the ground-truth label for the image located at `image_path`.
        In some datasets image ids are only unique per `subset`, i.e. validation
        or train subset.
        In this case, the subset must be specified.
        Raises a `KeyError` if no label is known for `image path`.
        """
        pass
