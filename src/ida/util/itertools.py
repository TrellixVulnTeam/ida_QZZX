from typing import Any, Optional

import numpy as np


def random_sublists(rng: np.random.Generator,
                    the_list: [Any],
                    max_size: Optional[int] = None,
                    include_empty: bool = False) -> [Any]:
    while True:
        sublist = []
        while len(sublist) == 0:
            all_idx = rng.random(len(the_list)) > .5
            if max_size is not None:
                selected_idx = set(rng.choice(range(len(the_list)), max_size))
            else:
                selected_idx = set(range(len(the_list)))
            sublist = [item for item_no, item in enumerate(the_list) if all_idx[item_no] and item_no in selected_idx]
            if include_empty:
                break
        yield sublist
