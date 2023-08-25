import copy
import numpy as np
from wfc.utils import GridCell, Direction, print_progress_bar
from collections import deque
import matplotlib.pyplot as plt

class Generator:
    ''' 
    Class which generates image by WFC algorithm using tiles list and their adjacency rules prepared by Preprocessor
    '''

    def __init__(self, output_dim, pixel_size, window_size, tiles, adjacency_rules, retry_num):
        # output dimensions in pixel_size
        self.width = output_dim[1]
        self.height = output_dim[0]
        # pixels can consist of multiple pixels
        self.pixel_size = pixel_size
        # window size in pixel_size
        self.window_size = window_size
        # list of all possible tiles from preprocessor
        self.tiles = tiles
        # tiles adjacency rules from preprocessor
        self.adjacency_rules = adjacency_rules
        # number of retries in case of contradictions
        self.retry_num = retry_num
        # final image pixel grid, will be intilized later
        self.grid = []
        
        self.direction2step = {Direction.BOTTOM: -self.width,
                               Direction.TOP: self.width,
                               Direction.LEFT: 1,
                               Direction.RIGHT: -1}

        # dict with structure tile_id: number of enablers in each direction
        # will be used to identify tiles with no enablers
        self.possible_tiles = {i: [len(adjacency_rules[Direction.BOTTOM][i]),
                               len(adjacency_rules[Direction.LEFT][i]),
                               len(adjacency_rules[Direction.TOP][i]),
                               len(adjacency_rules[Direction.RIGHT][i])]\
                            for i in range(len(tiles))}

        # cache values for quick grid intialization
        W = np.array([t.count for t in self.tiles])
        self.W_sum = np.sum(W)
        self.log_sum = np.sum(W * np.log(W))
        self.initial_entropy = np.log(self.W_sum) - self.log_sum/self.W_sum

        # flag indicating if there was a contradiction
        self.start_over = False
        # here info about deleted tiles will be stored
        self.updates = deque()
        self._reset()

    def generate(self):
        for i in range(1, self.retry_num+1):
            print(f"Generating image: attempt {i}/{self.retry_num}")

            count = 0
            while True:
                count += 1
                print_progress_bar(count, len(self.grid))

                # choose cell with minimal entropy
                chosen_cell = self._choose_cell()
                # if no such cell - all were already collapsed - end algorithm
                if chosen_cell is None:
                    break
                
                # collapse chosen cell
                self._collapse(chosen_cell)

                while self.updates:
                    update = self.updates.popleft()
                    self._propagate(*update)

                    # if there was a contradiction during update propagation - end this generation try
                    if self.start_over:
                        break
                
                if self.start_over:
                    print(f"Attempt {i} failed at {count/len(self.grid) * 100:0.0f}%")
                    break
            
            # if reached this place and no need to start over - end algorithm
            if not self.start_over:
                break
            
            if i < self.retry_num:
                self._reset()
        
        if not self.start_over:
            print("Image generated successfully")
            return self._prepare_image()
        
        print("All attempts failed, please retry")
        return None

    def _propagate(self, cell_id, tile_id):
        if self.start_over:
            return
        
        for direction in Direction:
            if not self._has_neighbour(cell_id, direction):
                continue

            adjacent_tiles = self.adjacency_rules[direction][tile_id]
            neighbour_cell = self.grid[cell_id + self.direction2step[direction]]

            for t in adjacent_tiles:
                # if tile was already deleted from neighbouring cell - no need to check it
                if not t in neighbour_cell.possible_tiles:
                    continue

                neighbour_cell.possible_tiles[t][direction] -= 1
                if neighbour_cell.possible_tiles[t][direction] == 0:
                    # if tile from neighbouring cell has no enablers - delete it
                    neighbour_cell.possible_tiles.pop(t)
                    
                    # if no tiles left after deletion - start over
                    if not neighbour_cell.possible_tiles:
                        self.start_over = True
                        return
                    
                    # if there possible tiles left - update neighbouring cells' entropy
                    self._update_entropy(neighbour_cell, t)
                    
                    # add info about tile deletion to updates queue
                    self.updates.append((neighbour_cell.id, t))

    def _collapse(self, cell):
        # construct tile distribution based on their counts
        probs = np.array([self.tiles[t].count for t in cell.possible_tiles])
        probs = probs/np.sum(probs)

        # choose tile randomly
        idx = np.random.choice(list(cell.possible_tiles.keys()), size=1, p=probs)[0]

        # delete all other tiles from cell
        for tile_id in cell.possible_tiles:
            if tile_id != idx:
                self.updates.append((cell.id, tile_id))

        cell.possible_tiles = {idx: cell.possible_tiles[idx]}
        # ensure cell won't be picked for collapse again
        # by setting its' entropy to highest possible value
        cell.entropy = np.inf

    # checks if cell has neighbour in direction
    def _has_neighbour(self, cell_id, direction):

        neighbour_idx = cell_id + self.direction2step[direction]

        if direction == Direction.BOTTOM:
            criterion = cell_id - self.width >= 0
        elif direction == Direction.TOP:
            criterion = cell_id + self.width < len(self.grid)
        elif direction == Direction.RIGHT:
            criterion = cell_id % self.width != 0
        else:
            criterion = (cell_id + 1) % self.width != 0

        return criterion and not np.isinf(self.grid[neighbour_idx].entropy)

    # updates cell entropy after deleted_tile deletion
    def _update_entropy(self, cell, deleted_tile):
        count = self.tiles[deleted_tile].count
        cell.W_sum -= count
        cell.log_sum -= count * np.log(count)

        cell.entropy = np.log(cell.W_sum) - cell.log_sum/cell.W_sum

    # samples cell with minimal entropy
    # if all cells were collapsed(have entropy infinity) returns None
    def _choose_cell(self):
        chosen_cell = min(self.grid, key=lambda cell: cell.entropy)
        if np.isinf(chosen_cell.entropy):
            return None

        return chosen_cell

    # resets grid, start_over flague and updates queue to initial state
    def _reset(self):
        print("Preparing for image generation")
        self.grid = [GridCell(copy.deepcopy(self.possible_tiles), 
                              i,
                              self.initial_entropy + np.random.normal(0, 1/2), 
                              self.W_sum, 
                              self.log_sum) for i in range(self.width*self.height)]

        self.start_over = False
        self.updates = deque()

    # constructs image after all cells were assigned a tile
    def _prepare_image(self):
        # init result image
        im = np.zeros((self.height * self.pixel_size, self.width * self.pixel_size, 3), dtype=int)

        for t in range(len(self.grid)):
            i = t // self.width
            j = t % self.width

            # pick tile assigned to this pixel
            idx = list(self.grid[t].possible_tiles.keys())[0]
            # pick right topmost pixel of this tile
            sub_tile = self.tiles[idx].tile[:self.pixel_size,
                                            (self.window_size - 1)*self.pixel_size:,
                                            :]
            # assign the value of this pixel to result image pixel
            im[i*self.pixel_size:(i+1)*self.pixel_size, 
               j*self.pixel_size:(j+1)*self.pixel_size, 
               :] = sub_tile

        return im



