import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def flip_axis_multi(images: xr.DataArray, dim: str, is_random=False) -> xr.DataArray:
    """Flip the axes of multiple images together.

    Such as flip left and right, up and down, randomly or non-randomly,

    Args:
        x (xr.DataArray): The array of images to flip.
        dim (str): The dimension to flip, typiclaly 'row' or 'col'.
        is_random (bool, optional): If applying or not the flipping should be random.
            Defaults to False.

    Returns:
        _type_: _description_
    """
    if is_random and np.random.uniform(-1, 1) <= 0:
        return images

    logger.info(f"Flipping dimension: {dim}.")
    flip = images.copy()
    flip.data[...] = flip.isel({dim: slice(None, None, -1)}).data
    return flip
