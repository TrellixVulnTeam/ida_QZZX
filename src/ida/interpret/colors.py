from typing import Optional, Tuple, Iterable

import numpy as np
import skimage

from ida.interpret.common import Interpreter


class PerceivableColorsInterpreter(Interpreter):
    """
    Describes each image with a set of "color masks".
    Each color mask represents all pixels of one perceivable color.
    """

    # names of all perceivable colors used by this describer
    COLOR_NAMES = ['red', 'orange', 'gold', 'yellow',
                   'green', 'turquoise', 'blue',
                   'purple', 'magenta',
                   'black', 'white', 'grey', 'unclear']

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

    def __init__(self,
                 random_state: int = 42,
                 max_concept_overlap: float = .4,
                 max_perturbed_area: float = .6):
        super().__init__(random_state=random_state,
                         max_concept_overlap=max_concept_overlap,
                         max_perturbed_area=max_perturbed_area)
        self._hue_bin_names = np.asarray(list(self.HUE_MAP.values()))
        self._hue_bins = np.array([0.] + list(self.HUE_MAP.keys())) / 360.
        self._pool = None

    def __str__(self) -> str:
        return 'perceivable_colors'

    @property
    def concepts(self) -> [str]:
        return self.COLOR_NAMES

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

    def __call__(self, image: Optional[np.ndarray], image_id: Optional[str], **kwargs) \
            -> Iterable[Tuple[int, np.ndarray]]:
        with np.errstate(divide='ignore', invalid='ignore'):
            hue, sat, light = self._rgb_to_hsl(image)

        maps = {'black': light < .1}
        remaining = np.logical_not(maps['black'])

        maps['white'] = np.bitwise_and(remaining, light > .9)
        remaining = np.bitwise_and(remaining, np.logical_not(maps['white']))

        grey_or_white_map = np.bitwise_and(remaining, sat < 0.1)
        maps['grey'] = np.bitwise_and(grey_or_white_map, light > 0.85)
        maps['white'] = np.bitwise_or(maps['white'],
                                      np.bitwise_and(grey_or_white_map, np.bitwise_not(maps['grey'])))
        remaining = np.bitwise_and(remaining, np.logical_not(grey_or_white_map))

        maps['unclear'] = np.bitwise_and(remaining, sat < 0.7)  # color of pixel is undefined, not clear enough
        remaining = np.bitwise_and(remaining, np.logical_not(maps['unclear']))

        hue_maps = self._hue_bins[:, None, None] > hue
        for hue_map, hue_name in zip(hue_maps, self._hue_bin_names):
            hue_map = np.bitwise_and(hue_map, remaining)
            if hue_name == 'red' and 'red' in maps:
                maps['red'] = np.bitwise_or(maps['red'], hue_map)
            else:
                maps[hue_name] = hue_map
            remaining = np.bitwise_and(remaining, np.logical_not(hue_map))

        maps['red'] = np.bitwise_or(maps['red'], remaining)  # the remaining are pixels with hue 1 = max. red

        # convert to integer ids and throw away empty masks
        return ((self.concepts.index(color_name), mask) for color_name, mask in maps.items() if np.any(mask))
