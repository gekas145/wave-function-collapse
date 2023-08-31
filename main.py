import numpy as np
import matplotlib.pyplot as plt
import cv2
from wfc.preprocessor import Preprocessor
from wfc.generator import Generator
from wfc.utils import Direction, GridCell

pixel_size = 8
window_size = 3
output_shape = (80, 80)
retry_num = 5
max_count = 250
input_im_path = "images/flowers.png"
output_im_path = "generated_images/flowers.png"

prep = Preprocessor(pixel_size, window_size, max_count=max_count)
prep.preprocess(input_im_path)
print(f"Tiles number: {len(prep.tiles)}")

gen = Generator(output_shape, pixel_size, window_size, prep.tiles, prep.adjacency_rules, retry_num)
im = gen.generate()

if im is not None:
    cv2.imwrite(output_im_path, im)

print("Image saved")

