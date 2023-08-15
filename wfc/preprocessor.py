import numpy as np
import cv2
from wfc.utils import Tile, Direction

class Preprocessor:

    def __init__(self, pixel_size, window_size):
        self.pixel_size = pixel_size
        self.window_size = window_size

        self.tiles = []
        self.adjacency_rules = []

    
    def _preprocess_tiles(self, im_path):
        im = cv2.imread(im_path)[:, :, ::-1]

        steps_w, steps_h = im.shape[1] // self.pixel_size, im.shape[0] // self.pixel_size

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

    
    def _preprocess_adjacency_rules(self):
        
        self.adjacency_rules = [np.ones((len(self.tiles), len(self.tiles)), dtype=bool),
                                np.ones((len(self.tiles), len(self.tiles)), dtype=bool)]

        for i in range(len(self.tiles)):
            for j in range(len(self.tiles)):
                for direction in [Direction.TOP, Direction.RIGHT]:
                    self.adjacency_rules[direction][i, j] = self.tiles[i].is_compatible(self.tiles[j], 
                                                                                        direction)
    
    def _augment(self, tile):
        tile_ref_hor = np.copy(tile[::-1, :, :])
        tile_ref_ver = np.copy(tile[:, ::-1, :])
        augmented = [tile, tile_ref_hor, tile_ref_ver]

        for i in range(3):
            tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            tile_ref_hor = cv2.rotate(tile_ref_hor, cv2.ROTATE_90_CLOCKWISE)
            tile_ref_ver = cv2.rotate(tile_ref_ver, cv2.ROTATE_90_CLOCKWISE)

            augmented += [tile, tile_ref_hor, tile_ref_ver]
        
        return augmented


