import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

from simple_parsing import ArgumentParser

from simadv.common import LoggingConfig
from simadv.places365.hub import Places365Hub
from simadv.util.webcache import WebCache


@dataclass
class ReorganizeTask:

    images_dir: Union[str, Path]
    cache_dir: Optional[str]
    glob = '*.jpg'
    subset: str = 'validation'

    def __post_init__(self):
        if isinstance(self.images_dir, str):
            self.images_dir = Path(self.images_dir)
        self.images_dir = self.images_dir.expanduser()
        assert self.images_dir.exists()

        if self.subset == 'validation':
            self.label_map = self.hub.validation_label_map
        elif self.subset == 'train':
            self.label_map = self.hub.train_label_map
        else:
            raise NotImplementedError('Currently only images from the '
                                      'validation or train subset can be '
                                      'reorganized.')

        if self.output_url is None:
            self.output_url = 'file://' + str(self.images_dir.absolute() /
                                              '{}.parquet'.format(self.subset))

        self.hub = Places365Hub(WebCache(self.cache_dir))

    def create_directories(self):
        """
        Creates a subdirectory for every Places365 label within `images_dir`.
        """
        for label in self.hub.label_names:
            label_path = self.images_dir / label
            label_path.mkdir(exist_ok=True)

    def run(self):
        """
        Moves all files matching `glob` within `images_dir` to the correct
        subdirectory. The correct subdirectory for an image has the name
        of the label of this image.
        Currently, this method only works for images from the validation subset.
        """
        for image_path in self.images_dir.glob(self.glob):
            label_id = self.label_map[image_path.name]
            dest_path = self.images_dir / self.hub.label_names[label_id] / \
                image_path.name
            logging.info('Moving image {} to {}.'.format(image_path, dest_path))
            image_path.replace(dest_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Group images by their label into different subdirectories.')
    parser.add_arguments(ReorganizeTask, dest='reorganize_task')
    parser.add_arguments(LoggingConfig, dest='logging')
    args = parser.parse_args()
    args.convert_task.run()
