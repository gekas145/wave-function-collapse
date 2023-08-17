import numpy as np
import matplotlib.pyplot as plt
import cv2
from wfc.preprocessor import Preprocessor
from wfc.generator import Generator
from wfc.utils import Direction, GridCell

pixel_size = 16
window_size = 3

prep = Preprocessor(pixel_size, window_size)

prep._preprocess_tiles("images/lines.png")
prep._preprocess_adjacency_rules()
print(len(prep.tiles))

d = Direction.LEFT
t = 5
print(prep.adjacency_rules[d][t])

plt.imshow(prep.tiles[t].tile)
plt.title("Chosen")
plt.show()

for i in prep.adjacency_rules[d][t]:
    plt.imshow(prep.tiles[i].tile)
    plt.title(i)
    plt.show()

# gen = Generator((5, 5), pixel_size, window_size, prep.tiles, prep.adjacency_rules)
# gen.generate()
# for g in gen.grid:
#     print(g.possible_tiles)
# im = gen.generate()
# if im is not None:
#     plt.imshow(im)
#     plt.show()

