import numpy as np
import matplotlib.pyplot as plt
import cv2
from wfc.preprocessor import Preprocessor

prep = Preprocessor(8, 4)

prep._preprocess_tiles("images/test-image.png")

for tile in prep.tiles:
    plt.imshow(tile.tile)
    plt.title(tile.count)
    plt.show()





