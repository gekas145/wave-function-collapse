import numpy as np
import matplotlib.pyplot as plt
import cv2
from wfc.preprocessor import Preprocessor
from wfc.generator import Generator
from wfc.utils import Direction, GridCell

prep = Preprocessor(15, 3)

prep._preprocess_tiles("images/flowers.png")
prep._preprocess_adjacency_rules()

gen = Generator((8, 8), 15, 3, prep.tiles, prep.adjacency_rules)
# gen.generate()
# for g in gen.grid:
#     print(g.possible_tiles)
im = gen.generate()
if im is not None:
    plt.imshow(im)
    plt.show()

