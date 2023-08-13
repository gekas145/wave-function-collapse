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

    def __init__(self, tiles, adjacency_rules):
        self.possible_tiles = list(range(len(tiles)))
        self.tiles = tiles
        self.adjacency_rules = adjacency_rules
        self.entropy = 0

        self.update_entropy()

    def _update_entropy(self):
        W = np.array([self.tiles[t].count for t in self.possible_tiles])
        W_sum = np.sum(W)

        self.entropy = np.log(W_sum) - np.sum(W * np.log(W))/W_sum

    def _compare_tiles(self, tile1, tile2, direction):
        if direction > 1:
            tile1, tile2 = tile2, tile1
    
        return self.adjacency_rules[direction % 2][tile1, tile2]

    def _check_enablers(self, tile, enablers, direction):
        for en in enablers:
            if self._compare_tiles(tile, en, direction):
                return True

        return False

    def check(self, tiles, direction):
        if np.isinf(self.entropy):
            return GridCellUpdate.UNCHANGED

        prev_len = len(self.possible_tiles)
        self.possible_tiles = [pt for pt in self.possible_tiles \
                               if self._check_enablers(pt, tiles, direction)]

        if prev_len != len(self.possible_tiles):
            if len(self.possible_tiles) == 0:
                return GridCellUpdate.CONTRADICTION
            
            self._update_entropy()

            return GridCellUpdate.CHANGED
        
        return GridCellUpdate.UNCHANGED




