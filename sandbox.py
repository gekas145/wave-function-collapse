import numpy as np
import matplotlib.pyplot as plt
import cv2
from wfc.preprocessor import Preprocessor
from wfc.generator import Generator
from wfc.utils import Direction, GridCell

pixel_size = 20
window_size = 3

prep = Preprocessor(pixel_size, window_size)

prep._preprocess_tiles("images/flowers.png")
prep._preprocess_adjacency_rules()

gen = Generator((5, 5), pixel_size, window_size, prep.tiles, prep.adjacency_rules)
# gen.generate()
# for g in gen.grid:
#     print(g.possible_tiles)
im = gen.generate()
if im is not None:
    plt.imshow(im)
    plt.show()

