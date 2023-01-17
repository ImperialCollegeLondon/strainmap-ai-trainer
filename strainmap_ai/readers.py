import logging
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def read_files(filenames: Union[str, List[str], List[Path]]) -> xr.DataArray:
    data = xr.open_mfdataset(filenames, combine="nested", concat_dim="frame")[
        "stacked"
    ].transpose("frame", "row", "col", "comp")
    logger.info(f"Input data array shape: {data.shape}")
    return data


def load_data(filename: Path, test_patients: Tuple = ()) -> xr.DataArray:
    """Loads data from NetCDF files, splitting them into train and test.

    Args:
        filename: Base directory where to start looking for NetCDF files.
        test_patients: Tuple with the initials of the patients to use for testing.

    Returns:
        Tuple with the datarrays of the loaded data, one for test and another for train.
    """
    if filename.is_dir():
        filenames = list(filename.glob("**/*_train.nc"))
    else:
        raise ValueError("'filename' must be the path to a directory.")

    test_filenames = []
    train_filenames = []
    if len(test_patients) == 0:
        train_filenames = filenames
    else:
        for f in filenames:
            for name in test_patients:
                if name in str(f):
                    test_filenames.append(f)
                else:
                    train_filenames.append(f)

    breakpoint()
    return read_files(filenames)


if __name__ == "__main__":

    filenames = Path(__file__).parent.parent.parent.parent / "Data"
    print(filenames)
    data = load_data(filenames)
    print(np.array(data.sel(comp=["MAG", "X", "Y", "Z"])).shape)
    print(np.array(data.sel(comp="LABELS")).shape)
