import numpy as np
from wfc.utils import GridCell, GridCellUpdate

class Generator:

    def __init__(self, output_dim, tiles, adjacency_rules):
        self.width = output_dim[1]
        self.height = output_dim[0]
        self.grid = [GridCell(tiles, adjacency_rules) for i in range(self.width*self.height)]

        self.start_over = False



    def generate(self):
        while True:
            min_entropy = min(self.grid, key=lambda x: x.entropy)
            if np.isinf(min_entropy):
                break

            min_entropy_cells = [cell for cell in self.grid if cell.entropy == min_entropy]
            collapsed_cell = np.random.choice(min_entropy_cells, size=1)[0]


    def _propagate(self):
        pass

    def _collapse(self, cell):
        pass

    def _on_grid(i, j):
        return 0 <= i < len(self.grid) and 0 <= j < len(self.grid[0])