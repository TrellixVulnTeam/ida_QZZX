import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional

from simple_parsing import ArgumentParser

from ida.common import LoggingConfig
from ida.places365.metadata import Places365MetadataProvider, Places365Task


@dataclass
class ReorganizeTask:
    """
    Moves all files matching `glob` within `images_dir` to the correct subdirectory.
    The correct subdirectory for an image has the name of the label of this image.
    Currently, this method only works for images from the validation and train subsets.
    """

    images_dir: Union[str, Path]
    cache_dir: Optional[str]
    glob = '*.jpg'
    subset: str = 'validation'
    task: Places365Task = field(default_factory=Places365Task, init=False)
    meta: Places365MetadataProvider = field(default_factory=Places365MetadataProvider, init=False)

    def __post_init__(self):
        if isinstance(self.images_dir, str):
            self.images_dir = Path(self.images_dir)
        self.images_dir = self.images_dir.expanduser()
        assert self.images_dir.exists()

        if self.subset not in ('validation', 'train'):
            raise NotImplementedError('Currently only images from the validation or train subset can be reorganized.')

        if self.output_url is None:
            self.output_url = 'file://' + str(self.images_dir.absolute() /
                                              '{}.parquet'.format(self.subset))

    def create_directories(self):
        """
        Creates a subdirectory for every Places365 label within `images_dir`.
        """
        for label in self.task.class_names:
            label_path = self.images_dir / label
            label_path.mkdir(exist_ok=True)

    def run(self):
        for image_path in self.images_dir.glob(self.glob):
            class_id = self.meta.get_image_class(image_path.name)
            dest_path = self.images_dir / self.task.class_names[class_id] / image_path.name
            logging.info('Moving image {} to {}.'.format(image_path, dest_path))
            image_path.replace(dest_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Group images by their label into different subdirectories.')
    parser.add_arguments(ReorganizeTask, dest='reorganize_task')
    parser.add_arguments(LoggingConfig, dest='logging')
    args = parser.parse_args()
    args.reorganize_task.run()
