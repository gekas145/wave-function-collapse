import numpy as np
import matplotlib.pyplot as plt
import cv2
from wfc.preprocessor import Preprocessor
from wfc.generator import Generator
from wfc.utils import Direction, GridCell

pixel_size = 10
window_size = 3
output_shape = (40, 150)
retry_num = 5

prep = Preprocessor(pixel_size, window_size)
prep.preprocess("images/knot.png")
print(f"Tiles number: {len(prep.tiles)}")

gen = Generator(output_shape, pixel_size, window_size, prep.tiles, prep.adjacency_rules, retry_num)
im = gen.generate()

if im is not None:
    cv2.imwrite("generated_images/knot.png", im)

print("Image saved")

