import numpy as np
import matplotlib.pyplot as plt
import cv2
from wfc.preprocessor import Preprocessor
from wfc.utils import Direction

prep = Preprocessor(8, 4)

prep._preprocess_tiles("images/test-image.png")

# for tile in prep.tiles[:2]:
#     plt.imshow(tile.tile[:, :, :])
#     plt.title(tile.count)
#     plt.show()



print(prep.tiles[0].is_compatible(prep.tiles[1], Direction.RIGHT))


