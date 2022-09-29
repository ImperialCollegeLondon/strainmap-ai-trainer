import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .readers import load_data
from .unet import DataAugmentation, UNet

logger = logging.getLogger(__name__)


def train(filenames: Path, model_file: Optional[Path] = None) -> None:

    # Data compiling and loading
    # read CSV file
    logger.info("Starting data loading...")
    data = load_data(filenames)

    logger.info("Data loading complete! Starting pre-processing...")
    augmented = DataAugmentation.factory().augment(data)
    logger.info("Data pre-processing complete!")

    # TODO: Normalisation step

    # Model training
    model = UNet.factory()
    model.compile_model()
    labels = np.array(augmented.sel(comp="LABELS").data)
    images = np.array(augmented.sel(comp=["MAG", "X", "Y", "Z"]).data)
    model.train(images, labels, model_file)
    logger.info(f"Training complete! Teained model saved to {model_file}")
