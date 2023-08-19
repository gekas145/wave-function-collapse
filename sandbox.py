import numpy as np
import matplotlib.pyplot as plt
import cv2
from wfc.preprocessor import Preprocessor
from wfc.generator import Generator
from wfc.utils import Direction, GridCell

pixel_size = 10
window_size = 3

prep = Preprocessor(pixel_size, window_size)

prep._preprocess_tiles("images/flowers.png")
prep._preprocess_adjacency_rules()
print(len(prep.tiles))

# for d in Direction:
#     print(prep.adjacency_rules[d])
#     print("=============")

# d = Direction.RIGHT
# t = 0
# print(prep.adjacency_rules[d][t])

# plt.imshow(prep.tiles[t].tile)
# plt.title("Chosen")
# plt.show()

# for i in prep.adjacency_rules[d][t]:
#     plt.imshow(prep.tiles[i].tile)
#     plt.title(i)
#     plt.show()

# tile_shape = prep.tiles[t].tile.shape[0]
# for tile in prep.tiles:
#     f, axarr = plt.subplots(1,2)
#     axarr[0].imshow(prep.tiles[t].tile[pixel_size:, :, :]) 
#     axarr[1].imshow(tile.tile[:tile_shape - pixel_size, :, :])
#     plt.show()

gen = Generator((20, 20), pixel_size, window_size, prep.tiles, prep.adjacency_rules)
# # gen.generate()
# # for g in gen.grid:
#     # print(g.possible_tiles)
im = gen.generate()
if im is not None:
    plt.imshow(im)
    plt.show()
# for g in gen.grid:
#     print(g.possible_tiles)

