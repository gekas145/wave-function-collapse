import numpy as np
import cv2
from wfc.utils import Tile, Direction
import matplotlib.pyplot as plt

class Preprocessor:
    ''' 
    Class which generates tiles list and their adjacency rules from input image for WFC algorithm
    '''

    def __init__(self, pixel_size, window_size):
        self.pixel_size = pixel_size
        self.window_size = window_size

        self.tiles = []
        self.adjacency_rules = []

    def preprocess(self, im_path):
        print("Preparing tiles")
        self._preprocess_tiles(im_path)
        print("Preparing adjacency rules")
        self._preprocess_adjacency_rules()

    def _preprocess_tiles(self, im_path):
        im = cv2.imread(im_path)

        steps_w, steps_h = im.shape[1] // self.pixel_size, im.shape[0] // self.pixel_size
        im = im[:steps_h*self.pixel_size, :steps_w*self.pixel_size, :]

        # enlarge the input image by wrapping it around
        A = im[:, 0:(self.window_size - 1)*self.pixel_size, :]
        B = im[0:(self.window_size - 1)*self.pixel_size, :, :]
        C = im[0:(self.window_size - 1)*self.pixel_size, 0:(self.window_size - 1)*self.pixel_size, :]

        im = np.concatenate((im, A), axis=1)
        B = np.concatenate((B, C), axis=1)
        im = np.concatenate((im, B), axis=0)

        tiles = []
        for i in range(steps_w):
            for j in range(steps_h):
                tile = im[j*self.pixel_size:j*self.pixel_size + self.window_size*self.pixel_size, 
                          i*self.pixel_size:i*self.pixel_size + self.window_size*self.pixel_size, 
                          :]
                tiles += self._augment(tile)

        tiles, counts = np.unique(tiles, return_counts=True, axis=0)
        self.tiles = [Tile(tile, count, self.pixel_size) for tile, count in zip(tiles, counts)]

    # generates adjacency rules in form that
    # adjacency_rules[direction[tile_id]] - is a list of all tiles ids which are compatible
    # with tiled_id in direction
    def _preprocess_adjacency_rules(self):
        
        self.adjacency_rules = [[[] for i in range(len(self.tiles))] for direction in Direction]

        for i in range(len(self.tiles)):
            for j in range(len(self.tiles)):
                if self.tiles[i].is_compatible(self.tiles[j], Direction.TOP):
                    self.adjacency_rules[Direction.TOP][i].append(j)
                    self.adjacency_rules[Direction.BOTTOM][j].append(i)
                
                if self.tiles[i].is_compatible(self.tiles[j], Direction.RIGHT):
                    self.adjacency_rules[Direction.RIGHT][i].append(j)
                    self.adjacency_rules[Direction.LEFT][j].append(i)

                
    # augments tile by generating all its' rotations and their reflections
    def _augment(self, tile):
        tile_ref_hor = np.copy(tile[::-1, :, :])
        tile_ref_ver = np.copy(tile[:, ::-1, :])
        result = [tile, tile_ref_hor, tile_ref_ver]

        for i in range(3):
            tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            tile_ref_hor = cv2.rotate(tile_ref_hor, cv2.ROTATE_90_CLOCKWISE)
            tile_ref_ver = cv2.rotate(tile_ref_ver, cv2.ROTATE_90_CLOCKWISE)

            result += [tile, tile_ref_hor, tile_ref_ver]
        
        return result


