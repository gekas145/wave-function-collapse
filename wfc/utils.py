import numpy as np
from enum import IntEnum


class Direction(IntEnum):

    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

class Tile:

    def __init__(self, tile, count, pixel_size):
        self.tile = tile
        self.count = count
        self.pixel_size = pixel_size
    
    # checks compatibility in top and right directions
    # compatibility in other directions can be obtained by 
    # calling this method on other with self as argument
    def is_compatible(self, other, direction):
        tile_shape = self.tile.shape[0]

        if direction == Direction.TOP:
            return np.array_equal(self.tile[self.pixel_size:, :, :],
                                  other.tile[:tile_shape - self.pixel_size, :, :])

        # else check right adjacency
        return np.array_equal(self.tile[:, :tile_shape - self.pixel_size, :],
                              other.tile[:, self.pixel_size:, :])


# stores all info about grid cell from output image grif in Generator
class GridCell:

    def __init__(self, possible_tiles, id, initial_entropy, W_sum, log_sum):
        # dict of form tiled_id: list of enablers number in each possible direction
        self.possible_tiles = possible_tiles
        self.id = id
        self.entropy = initial_entropy
        # cached values for efficient entropy updates
        self.W_sum = W_sum
        self.log_sum = log_sum

# this code was taken from
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console?page=1&tab=votes#tab-top
def print_progress_bar(iteration, total, decimals=0, length=10, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    end = "\r" if iteration != total else "\n"
    print(f'|{bar}| {percent}%', end=end)

