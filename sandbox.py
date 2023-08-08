import numpy as np
import matplotlib.pyplot as plt
import cv2



im = cv2.imread("images/test-image.png")[:, :, ::-1]

pixel_size = 8
window_size = 3

A = im[:, 0:(window_size - 1)*pixel_size, :]
B = im[0:(window_size - 1)*pixel_size, :, :]
C = im[0:(window_size - 1)*pixel_size, 0:(window_size - 1)*pixel_size, :]

im = np.concatenate((im, A), axis=1)
B = np.concatenate((B, C), axis=1)
im = np.concatenate((im, B), axis=0)

tiles = [im[j*pixel_size:j*pixel_size+3*pixel_size, i*pixel_size:i*pixel_size+3*pixel_size, :] \
         for i in range(8) for j in range(8)]

# for t in tiles:
#     print(t.shape)


tiles, counts = np.unique(tiles, return_counts=True, axis=0)

for i in range(len(tiles)):
    plt.imshow(tiles[i])
    plt.title(counts[i])
    plt.show()

print(tiles.shape)
print(counts)

plt.imshow(im)
plt.show()



