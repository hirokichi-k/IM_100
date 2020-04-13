import cv2
import numpy as np


def reduce(img):
    img[(img >= 0) & (img < 64)] = 32
    img[(img >= 64) & (img < 128)] = 96
    img[(img >= 128) & (img < 192)] = 160
    img[(img >= 192) & (img < 256)] = 224
    return img

img = cv2.imread('imori.jpg').astype(np.float32)

out = reduce(img)

cv2.imwrite("out_figure6.jpg",out)

