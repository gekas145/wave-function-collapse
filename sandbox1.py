import numpy as np
import time
import cv2
from collections import deque
import matplotlib.pyplot as plt

im = cv2.imread("images/flowers.png")
im = im
# im = []

a = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
a = cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)
# a *= 0

plt.imshow(im)
plt.show()