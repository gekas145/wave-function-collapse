import copy
import numpy as np
from wfc.utils import GridCell, Direction, print_progress_bar
from collections import deque
import matplotlib.pyplot as plt

class Generator:

    def __init__(self, output_dim, pixel_size, window_size, tiles, adjacency_rules, retry_num):
        self.width = output_dim[1]
        self.height = output_dim[0]
        self.pixel_size = pixel_size
        self.window_size = window_size
        self.tiles = tiles
        self.adjacency_rules = adjacency_rules
        self.retry_num = retry_num
        self.grid = {}

        self.direction2step = {Direction.BOTTOM: -self.width,
                               Direction.TOP: self.width,
                               Direction.LEFT: 1,
                               Direction.RIGHT: -1}

        self.possible_tiles = {i: [len(adjacency_rules[Direction.BOTTOM][i]),
                               len(adjacency_rules[Direction.LEFT][i]),
                               len(adjacency_rules[Direction.TOP][i]),
                               len(adjacency_rules[Direction.RIGHT][i])]\
                            for i in range(len(tiles))}

        W = np.array([t.count for t in self.tiles])
        self.W_sum = np.sum(W)
        self.log_sum = np.sum(W * np.log(W))

        self.initial_entropy = np.log(self.W_sum) - self.log_sum/self.W_sum

        self.start_over = False
        self.updates = deque()
        self._reset()

    def generate(self):
        for i in range(self.retry_num):
            print(f"Generating image: attempt {i+1}/{self.retry_num}")

            count = 0
            while True:
                count += 1
                print_progress_bar(count, len(self.grid))

                chosen_cell = self._choose_cell()
                if chosen_cell is None:
                    break

                self._collapse(chosen_cell)

                while self.updates:
                    update = self.updates.popleft()
                    self._propagate(*update)

                    if self.start_over:
                        break
                
                if self.start_over:
                    print(f"Attempt {i+1} failed")
                    break
        
            if not self.start_over:
                break

            self._reset()
        
        if not self.start_over:
            print("Image generated successfully")
            return self._prepare_image()
        
        print("All attepts failed, please retry")
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
                if not t in neighbour_cell.possible_tiles:
                    continue

                neighbour_cell.possible_tiles[t][direction] -= 1
                if neighbour_cell.possible_tiles[t][direction] == 0:
                    neighbour_cell.possible_tiles.pop(t)
                    
                    if not neighbour_cell.possible_tiles:
                        self.start_over = True
                        return
                    
                    self._update_entropy(neighbour_cell, t)
                    
                    self.updates.append((neighbour_cell.id, t))

    def _collapse(self, cell):
        probs = np.array([self.tiles[t].count for t in cell.possible_tiles])
        probs = probs/np.sum(probs)

        idx = np.random.choice(list(cell.possible_tiles.keys()), size=1, p=probs)[0]

        for tile_id in cell.possible_tiles:
            if tile_id != idx:
                self.updates.append((cell.id, tile_id))

        cell.possible_tiles = {idx: cell.possible_tiles[idx]}
        cell.entropy = np.inf

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

    def _update_entropy(self, cell, deleted_tile):
        count = self.tiles[deleted_tile].count
        cell.W_sum -= count
        cell.log_sum -= count * np.log(count)

        cell.entropy = np.log(cell.W_sum) - cell.log_sum/cell.W_sum

    def _choose_cell(self):
        chosen_cell = min(self.grid, key=lambda cell: cell.entropy)
        if np.isinf(chosen_cell.entropy):
            return None

        return chosen_cell

    def _reset(self):
        self.grid = [GridCell(copy.deepcopy(self.possible_tiles), 
                              i,
                              self.initial_entropy + np.random.normal(0, 1/2), 
                              self.W_sum, 
                              self.log_sum) for i in range(self.width*self.height)]

        self.start_over = False
        self.updates = deque()

    def _prepare_image(self):
        im = np.zeros((self.height * self.pixel_size, self.width * self.pixel_size, 3), dtype=int)

        for t in range(len(self.grid)):
            i = t // self.width
            j = t % self.width

            idx = list(self.grid[t].possible_tiles.keys())[0]
            sub_tile = self.tiles[idx].tile[:self.pixel_size,
                                            (self.window_size - 1)*self.pixel_size:,
                                            :]

            im[i*self.pixel_size:(i+1)*self.pixel_size, 
               j*self.pixel_size:(j+1)*self.pixel_size, 
               :] = sub_tile

        return im



