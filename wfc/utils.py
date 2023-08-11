from enum import Enum


class Direction(Enum):

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Tile:

    def __init__(self, tile, count, pixel_size):
        self.tile = tile
        self.count = count
        self.pixel_size = pixel_size

    def is_compatible(self, other, direction):
        pass