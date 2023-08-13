import numpy as np
import matplotlib.pyplot as plt
import cv2
from wfc.preprocessor import Preprocessor
from wfc.utils import Direction, GridCell

prep = Preprocessor(16, 3)

prep._preprocess_tiles("images/test-image.png")
prep._preprocess_adjacency_rules()

gc = GridCell(prep.tiles, prep.adjacency_rules)
print(gc.entropy)



# print(len(prep.tiles))

direction = Direction.BOTTOM
tile_num = 0

plt.imshow(prep.tiles[tile_num].tile)
plt.title("Left")
plt.show()

state = gc.check([tile_num], direction)
print(state)
print(gc.possible_tiles)
print(gc.entropy)

for pt in gc.possible_tiles:
    plt.imshow(gc.tiles[pt].tile)
    plt.title(pt)
    plt.show()

# for i in range(len(prep.tiles)):
#     if direction > 1:
#         check = prep.adjacency_rules[direction % 2][i, tile_num]
#     else:
#         check = prep.adjacency_rules[direction % 2][tile_num, i]

#     if check:
#         plt.imshow(prep.tiles[i].tile)
#         plt.title(f"Tile {i}")
#         plt.show()

# for tile in prep.tiles[:2]:
#     plt.imshow(tile.tile[:, :, :])
#     plt.title(tile.count)
#     plt.show()



# print(prep.tiles[0].is_compatible(prep.tiles[1], Direction.RIGHT))


