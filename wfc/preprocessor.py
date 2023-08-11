import numpy as np
import cv2
from wfc.utils import Tile

class Preprocessor:

    def __init__(self, pixel_size, window_size):
        self.pixel_size = pixel_size
        self.window_size = window_size

        self.tiles = []
        self.adjacency_rules = []

    
    def _preprocess_tiles(self, im_path):
        im = cv2.imread(im_path)[:, :, ::-1]

        A = im[:, 0:(self.window_size - 1)*self.pixel_size, :]
        B = im[0:(self.window_size - 1)*self.pixel_size, :, :]
        C = im[0:(self.window_size - 1)*self.pixel_size, 0:(self.window_size - 1)*self.pixel_size, :]

        im = np.concatenate((im, A), axis=1)
        B = np.concatenate((B, C), axis=1)
        im = np.concatenate((im, B), axis=0)

        tiles = [im[j*self.pixel_size:j*self.pixel_size + self.window_size*self.pixel_size, 
                    i*self.pixel_size:i*self.pixel_size + self.window_size*self.pixel_size, :] \
                for i in range(self.pixel_size) for j in range(self.pixel_size)]

        tiles, counts = np.unique(tiles, return_counts=True, axis=0)
        self.tiles = [Tile(tile, count, self.pixel_size) for tile, count in zip(tiles, counts)]

    
    def _preprocess_adjacency_rules():
        pass


