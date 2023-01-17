import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .readers import load_data
from .unet import DataAugmentation, Normal, UNet

logger = logging.getLogger(__name__)


def train(
    filenames: Path, model_file: Optional[Path] = None, test_patients: Tuple = ()
) -> None:
    crop = 64

    # Data compiling and loading
    # read CSV file
    logger.info("Starting data loading...")
    train, test = load_data(filenames, test_patients)
    max_rows = train.sizes["row"] - crop
    max_cols = train.sizes["col"] - crop
    train = train.sel(row=range(crop, max_rows), col=range(crop, max_cols))
    logger.info("... data loading complete!")
    rows = train.sizes["row"]
    cols = train.sizes["col"]

    logger.info("Starting pre-processing...")
    augmented = DataAugmentation.factory().augment(train)
    logger.info("... data pre-processing complete!")

    # Normalise the train data
    logger.info("Starting data normalisation...")
    labels_train = augmented.sel(comp="LABELS").data
    images_train = Normal().run(
        augmented.sel(comp=["MAG", "X", "Y", "Z"]).data, method="ubytes"
    )

    # Normalise the test data
    if len(test):
        test = test.sel(row=range(crop, max_rows), col=range(crop, max_cols))
        labels_test = test.sel(comp="LABELS").data
        images_test = Normal().run(
            test.sel(comp=["MAG", "X", "Y", "Z"]).data, method="ubytes"
        )
    else:
        labels_test = test
        images_test = test
    logger.info("... data normalisation complete!")

    # Model training
    logger.info("Starting AI training (this might take a while)...")
    model = UNet.factory()
    model.img_height = rows
    model.img_width = cols
    model.compile_model()
    model.train(
        np.asarray(images_train),
        np.asarray(labels_train),
        np.asarray(images_test),
        np.asarray(labels_test),
        model_file,
    )
    logger.info(f"Training complete! Trained model saved to: {model_file}")
