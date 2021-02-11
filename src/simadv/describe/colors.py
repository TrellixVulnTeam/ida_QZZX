from dataclasses import dataclass, field

import numpy as np
import skimage
from petastorm.unischema import Unischema

from simadv.spark import Field, Schema
from simadv.describe.common import DictBasedImageDescriber, ImageReadConfig


@dataclass
class PerceivableColorsImageDescriber(DictBasedImageDescriber):
    """
    Describes each image with a set of "color masks".
    Each color mask represents all pixels of one perceivable color.
    """

    read_cfg: ImageReadConfig
    output_schema: Unischema = field(default=Schema.CONCEPT_MASKS, init=False)

    name: str = field(default='perceivable_colors', init=False)

    # names of all perceivable colors used by this describer
    COLOR_NAMES = np.char.array(['red', 'orange', 'gold', 'yellow',
                                 'green', 'turquoise', 'blue',
                                 'purple', 'magenta',
                                 'black', 'white', 'grey'])

    # assign natural language color names to hue values
    HUE_MAP = {20.: 'red',
               45.: 'orange',
               55.: 'gold',
               65.: 'yellow',
               155.: 'green',
               185.: 'turquoise',
               250.: 'blue',
               280.: 'purple',
               320.: 'magenta',
               360.: 'red'}

    resize_to: int = 100

    def __post_init__(self):
        self._hue_bin_names = np.asarray(list(self.HUE_MAP.values()))
        self._hue_bins = np.array([0.] + list(self.HUE_MAP.keys())) / 360.
        self._pool = None

    @staticmethod
    def _rgb_to_hsl(rgb):
        rgb = skimage.img_as_float(rgb)
        minimum = np.amin(rgb, -1)
        maximum = np.amax(rgb, -1)
        delta = np.ptp(rgb, -1)

        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        light = (maximum + minimum) / 2

        sat = np.where(
            light < 0.5,
            delta / (maximum + minimum),
            delta / (2 - maximum - minimum),
        )
        sat[np.asarray(delta == 0)] = 0

        delta_r = (((maximum - r) / 6) + (delta / 2)) / delta
        delta_g = (((maximum - g) / 6) + (delta / 2)) / delta
        delta_b = (((maximum - b) / 6) + (delta / 2)) / delta

        hue = delta_b - delta_g
        hue = np.where(g == maximum, (1 / 3) + delta_r - delta_b, hue)
        hue = np.where(b == maximum, (2 / 3) + delta_g - delta_r, hue)
        hue[np.asarray(hue < 0)] += 1
        hue[np.asarray(hue > 1)] -= 1
        hue[np.asarray(delta == 0)] = 0

        return hue, sat, light

    def generate(self):
        with self.read_cfg.make_reader(None) as reader:
            for row in reader:
                with np.errstate(divide='ignore', invalid='ignore'):
                    hue, sat, light = self._rgb_to_hsl(row.image)

                maps = {'black': light < .1}
                remaining = np.logical_not(maps['black'])

                maps['white'] = np.bitwise_and(remaining, light > .9)
                remaining = np.bitwise_and(remaining, np.logical_not(maps['white']))

                grey_or_white_map = np.bitwise_and(remaining, sat < 0.1)
                maps['grey'] = np.bitwise_and(grey_or_white_map, light > 0.85)
                maps['white'] = np.bitwise_or(maps['white'],
                                              np.bitwise_and(grey_or_white_map, np.bitwise_not(maps['grey'])))
                remaining = np.bitwise_and(remaining, np.logical_not(grey_or_white_map))

                maps['none'] = np.bitwise_and(remaining, sat < 0.7)  # color of pixel is undefined, not clear enough
                remaining = np.bitwise_and(remaining, np.logical_not(maps['none']))

                hue_maps = self._hue_bins[:, None, None] > hue
                for hue_map, hue_name in zip(hue_maps, self._hue_bin_names):
                    hue_map = np.bitwise_and(hue_map, remaining)
                    if hue_name == 'red' and 'red' in maps:
                        maps['red'] = np.bitwise_or(maps['red'], hue_map)
                    else:
                        maps[hue_name] = hue_map
                    remaining = np.bitwise_and(remaining, np.logical_not(hue_map))

                maps['red'] = np.bitwise_or(maps['red'], remaining)  # the remaining are pixels with hue 1 = max. red

                yield {Field.IMAGE_ID.name: row.image_id,
                       Field.DESCRIBER.name: self.name,
                       Field.CONCEPT_NAMES.name: np.asarray(list(maps.keys())),
                       Field.CONCEPT_MASKS.name: np.asarray(list(maps.values()))}
