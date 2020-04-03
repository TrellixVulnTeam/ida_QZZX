import argparse
import logging
from pathlib import Path

from thesis_impl.places365.hub import Places365Hub
from thesis_impl.places365 import config as cfg
from thesis_impl.util.webcache import WebCache


class Reorganizer:

    def __init__(self, images_dir: str, hub: Places365Hub):
        self.images_dir = Path(images_dir)
        assert self.images_dir.exists()
        self.hub = hub

    def create_directories(self):
        """
        Creates a subdirectory for every Places365 label within `images_dir`.
        """
        for label in self.hub.all_labels:
            label_path = self.images_dir / label
            label_path.mkdir(exist_ok=True)

    def move_image_files(self, glob='*.jpg', subset='validation'):
        """
        Moves all files matching `glob` within `images_dir` to the correct
        subdirectory. The correct subdirectory for an image has the name
        of the label of this image.
        Currently, this method only works for images from the validation subset.
        """
        if subset == 'validation':
            label_map = self.hub.validation_label_map
        elif subset == 'train':
            label_map = self.hub.train_label_map
        else:
            raise NotImplementedError('Currently only images from the '
                                      'validation or train subset can be '
                                      'reorganized.')

        for image_path in self.images_dir.glob(glob):
            label_id = label_map[image_path.name]
            dest_path = self.images_dir / self.hub.all_labels[label_id] / \
                        image_path.name
            logging.info('Moving image {} to {}.'.format(image_path, dest_path))
            image_path.replace(dest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group images by their label '
                                                 'into different '
                                                 'subdirectories.')
    parser.add_argument('subset', type=str, choices=['validation', 'train'],
                        help='the subset of images to reorganize')
    parser.add_argument('images_dir', type=str,
                        help='the directory where the images are stored')
    parser.add_argument('images_glob', type=str,
                        help='glob expression specifying which images in the '
                             'above directory should be moved')

    log_group = parser.add_argument_group('Logging settings')
    cfg.LoggingConfig.setup_parser(log_group)

    cache_group = parser.add_argument_group('Cache settings')
    cfg.WebCacheConfig.setup_parser(cache_group)

    args = parser.parse_args()

    cfg.LoggingConfig.set_from_args(args)
    _cache = WebCache(cfg.WebCacheConfig.from_args(args))
    _hub = Places365Hub(_cache)

    reorganizer = Reorganizer(args.images_dir, _hub)
    reorganizer.create_directories()
    reorganizer.move_image_files(args.images_glob, args.subset)
