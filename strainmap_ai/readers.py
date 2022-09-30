import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def read_files(filenames: Union[str, List[str], List[Path]]) -> xr.DataArray:
    data = xr.open_mfdataset(filenames, combine="nested", concat_dim="frame")[
        "stacked"
    ].transpose("frame", "row", "col", "comp")
    logger.info(f"Input data array shape: {data.shape}")
    return data


def filenames_from_csv(filename: Path) -> List[str]:
    ...


def load_data(filename: Path) -> xr.DataArray:
    if filename.suffix == "csv":
        filenames = filenames_from_csv(filename)
    elif filename.is_dir():
        filenames = str(filename / "**" / "*_train.nc")
    else:
        raise ValueError("'filename' must be the path to a CSV file or to a directory.")

    return read_files(filenames)


if __name__ == "__main__":

    filenames = Path(__file__).parent.parent.parent / "Data"
    print(filenames)
    data = load_data(filenames)
    print(np.array(data.sel(comp=["MAG", "X", "Y", "Z"])).shape)
    print(np.array(data.sel(comp="LABELS")).shape)
