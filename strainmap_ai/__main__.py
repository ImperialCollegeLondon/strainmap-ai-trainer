import argparse
from pathlib import Path

if __name__ == "__main__":
    from .scripts import train

    parser = argparse.ArgumentParser(
        prog="StrainMap AI Trainer", description="Train a new AI with segmented data."
    )
    parser.add_argument(
        "datapath",
        type=str,
        help="Path to start the search for data files, which must end in '_train.nc'.",
    )
    args = parser.parse_args()

    filenames = Path(__file__).parent.parent.parent / "Data"
    filenames = Path(args.datapath).resolve()
    print(filenames)
    train(filenames)
