import numpy as np
from wfc.utils import GridCell, GridCellUpdate, Direction
from collections import deque
import matplotlib.pyplot as plt

class Generator:

    def __init__(self, output_dim, pixel_size, window_size, tiles, adjacency_rules):
        self.width = output_dim[1]
        self.height = output_dim[0]
        self.pixel_size = pixel_size
        self.window_size = window_size
        self.grid = [GridCell(tiles, adjacency_rules, i) for i in range(self.width*self.height)]

        self.start_over = False
        self.updates = deque()
        self.direction2step = {Direction.TOP: -self.width,
                               Direction.BOTTOM: self.width,
                               Direction.RIGHT: 1,
                               Direction.LEFT: -1}



    def generate(self):
        count = 0
        while True:
            count += 1
            print(f"{count}/{len(self.grid)}")
            chosen_cell = self._choose_cell()
            if chosen_cell is None:
                break

            self._collapse(chosen_cell)
            self.updates.append((None, chosen_cell.id))

            while self.updates:
                update = self.updates.popleft()
                self._propagate(*update)

                if self.start_over:
                    break
            
            if self.start_over:
                    break
        
        if not self.start_over:
            return self._prepare_image()
        return None



    def _propagate(self, direction, cell_id):
        if (np.isinf(self.grid[cell_id].entropy) and direction is not None) or self.start_over:
            return
        
        if direction is not None:
            mod_cell_id = cell_id - self.direction2step[direction]
            state = self.grid[cell_id].check(self.grid[mod_cell_id].possible_tiles, 
                                             direction)
        else:
            state = GridCellUpdate.CHANGED

        if state == GridCellUpdate.UNCHANGED:
            return

        if state == GridCellUpdate.CONTRADICTION:
            self.start_over = True
            return
        
        if cell_id - self.width >= 0 and direction != Direction.BOTTOM:
            self.updates.append((Direction.TOP, 
                                 cell_id + self.direction2step[Direction.TOP]))

        if cell_id + self.width < len(self.grid) and direction != Direction.TOP:
            self.updates.append((Direction.BOTTOM, 
                                 cell_id + self.direction2step[Direction.BOTTOM]))

        if cell_id % self.width != 0 and direction != Direction.RIGHT:
            self.updates.append((Direction.LEFT, 
                                 cell_id + self.direction2step[Direction.LEFT]))

        if (cell_id + 1) % self.width != 0 and direction != Direction.LEFT:
            self.updates.append((Direction.RIGHT, 
                                 cell_id + self.direction2step[Direction.RIGHT]))

        

    def _collapse(self, cell):
        probs = np.array([cell.tiles[t].count for t in cell.possible_tiles])
        probs = probs/np.sum(probs)

        idx = np.random.choice(np.arange(len(cell.possible_tiles)), size=1, p=probs)[0]

        cell.possible_tiles = [cell.possible_tiles[idx]]
        cell.entropy = np.inf

    def _choose_cell(self):
        min_entropy = min(self.grid, key=lambda x: x.entropy).entropy
        if np.isinf(min_entropy):
            return None

        min_entropy_cells = [cell for cell in self.grid if cell.entropy == min_entropy]
        return np.random.choice(min_entropy_cells, size=1)[0]

    def _prepare_image(self):
        im = np.zeros((self.height * self.pixel_size, self.width * self.pixel_size, 3), dtype=int)
        tiles = self.grid[0].tiles

        for t in range(len(self.grid)):
            i = t // self.width
            j = t % self.width

            sub_tile = tiles[self.grid[t].possible_tiles[0]].tile[:self.pixel_size, 
                                                                  (self.window_size - 1)*self.pixel_size:, 
                                                                  :]          

            im[i*self.pixel_size:(i+1)*self.pixel_size, 
               j*self.pixel_size:(j+1)*self.pixel_size, 
               :] = sub_tile

        return im



