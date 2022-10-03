import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .readers import load_data
from .unet import DataAugmentation, Normal, UNet

logger = logging.getLogger(__name__)


def train(filenames: Path, model_file: Optional[Path] = None) -> None:
    crop = slice(100, -100, 1)

    # Data compiling and loading
    # read CSV file
    logger.info("Starting data loading...")
    data = load_data(filenames).sel(row=crop, col=crop)
    logger.info("... data loading complete!")

    logger.info("Starting pre-processing...")
    augmented = DataAugmentation.factory().augment(data)
    logger.info("... data pre-processing complete!")

    # Normalise the data
    logger.info("Starting data normalisation...")
    labels = augmented.sel(comp="LABELS").data
    images = Normal().run(
        augmented.sel(comp=["MAG", "X", "Y", "Z"]).data, method="ubytes"
    )
    logger.info("... data normalisation complete!")

    # Model training
    logger.info("Starting AI training (this might take a while)...")
    model = UNet.factory()
    model.img_height = images.sizes("row")
    model.img_width = images.sizes("col")
    model.compile_model()
    model.train(np.asarray(images), np.asarray(labels), model_file)
    logger.info(f"Training complete! Trained model saved to: {model_file}")
