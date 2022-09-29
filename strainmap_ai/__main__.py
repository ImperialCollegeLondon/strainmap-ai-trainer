import argparse
import logging
from pathlib import Path

logger = logging.getLogger("StrainMap_AI")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser to handle comand line arguments."""

    parser = argparse.ArgumentParser(
        prog="StrainMap AI Trainer", description="Train a new AI with segmented data."
    )
    parser.add_argument(
        "datapath",
        type=str,
        help="Path to start the search for data files, which must end in '_train.nc'.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path where the trained model should be saved (default =  None).",
    )
    return parser


if __name__ == "__main__":
    from .scripts import train

    args = create_parser().parse_args()

    # filenames = Path(__file__).parent.parent.parent / "Data"
    filenames = Path(args.datapath).resolve()
    model_path = Path(args.model_path).resolve() if args.model_path else None
    logger.info(f"Path where to search for data files: {filenames}")
    logger.info(f"Path where the trained model will be saved: {model_path}")

    train(filenames=filenames, model_file=model_path)
