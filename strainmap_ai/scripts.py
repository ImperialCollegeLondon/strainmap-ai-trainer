import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .readers import load_data
from .unet import DataAugmentation, Normal, UNet

logger = logging.getLogger(__name__)


def train(filenames: Path, model_file: Optional[Path] = None) -> None:

    # Data compiling and loading
    # read CSV file
    data = load_data(filenames)
    # labels = np.array(data.sel(comp="LABELS").data)
    # images = np.array(data.sel(comp=["MAG", "X", "Y", "Z"]).data)
    logger.info("Data loading complete! Starting pre-processing")

    # Data pre-processing
    # images = Normal.run(images, "zeromean_unitvar")
    augmented = DataAugmentation.factory().augment(data)
    logger.info("Data pre-processing complete!")

    # Model training
    model = UNet.factory()
    model.compile_model()
    labels = np.array(augmented.sel(comp="LABELS").data)
    images = np.array(augmented.sel(comp=["MAG", "X", "Y", "Z"]).data)
    model.train(images, labels, model_file)
    logger.info(f"Training complete! Teained model saved to {model_file}")
