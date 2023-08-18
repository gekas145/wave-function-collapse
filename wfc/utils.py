import numpy as np
from enum import IntEnum


class Direction(IntEnum):

    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

class GridCellUpdate(IntEnum):

    CHANGED = 0
    UNCHANGED = 1
    CONTRADICTION = 2


class Tile:

    def __init__(self, tile, count, pixel_size):
        self.tile = tile
        self.count = count
        self.pixel_size = pixel_size

    def is_compatible(self, other, direction):
        tile_shape = self.tile.shape[0]

        if direction == Direction.TOP:
            return np.array_equal(self.tile[self.pixel_size:, :, :],
                                  other.tile[:tile_shape - self.pixel_size, :, :])

        # else check right adjacency
        return np.array_equal(self.tile[:, :tile_shape - self.pixel_size, :],
                              other.tile[:, self.pixel_size:, :])



class GridCell:

    def __init__(self, possible_tiles, id):
        self.possible_tiles = possible_tiles
        self.id = id
        self.entropy = 0




