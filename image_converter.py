import cv2
from pathlib import Path
from strainmap_ai.readers import load_data
import numpy as np
from logging import getLogger

data = load_data(Path("../../Data").resolve())
output = Path("../Data/All").resolve()
val_fraction = 0.2
logger = getLogger()

for loc in ("val", "train"):
    (output / f"labels_{loc}" / "img").mkdir(parents=True, exist_ok=True)
    (output / f"images_{loc}" / "img").mkdir(parents=True, exist_ok=True)

print(f"Number of images: {len(data)}")
labels = np.asarray(data.sel(comp=["LABELS"]).data.astype(np.uint8))
images = np.asarray(data.sel(comp=["MAG", "X", "Y", "Z"]).data)
images = images / images.max()

for i in range(len(data)):
    loc = "val" if np.random.uniform(0, 1) <= val_fraction else "train"
    logger.info(f"Saving to '{output}/*_{loc}/img' image {i}")
    cv2.imwrite(str(output / f"labels_{loc}" / "img" / f"{i:05}.png"), labels[i])
    cv2.imwrite(
        str(output / f"images_{loc}" / "img" / f"{i:05}.png"),
        (images[i] / images[i].max() * np.iinfo(np.uint16).max).astype(np.uint16),
    )
