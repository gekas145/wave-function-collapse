import numpy as np
import matplotlib.pyplot as plt
import cv2
from wfc.preprocessor import Preprocessor
from wfc.utils import Direction

prep = Preprocessor(16, 3)

prep._preprocess_tiles("images/test-image.png")
prep._preprocess_adjacency_rules()

print(len(prep.tiles))

direction = Direction.BOTTOM
tile_num = 4

plt.imshow(prep.tiles[tile_num].tile)
plt.title("Chosen")
plt.show()



for i in range(len(prep.tiles)):
    if direction > 1:
        check = prep.adjacency_rules[direction % 2][i, tile_num]
    else:
        check = prep.adjacency_rules[direction % 2][tile_num, i]

    if check:
        plt.imshow(prep.tiles[i].tile)
        plt.show()

# for tile in prep.tiles[:2]:
#     plt.imshow(tile.tile[:, :, :])
#     plt.title(tile.count)
#     plt.show()



# print(prep.tiles[0].is_compatible(prep.tiles[1], Direction.RIGHT))


