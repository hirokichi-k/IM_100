import cv2
import numpy as np

def mean_pool(img):
    out = img.copy()
    g = 8
    H,W,C = img.shape
    H = int(H/g)
    W = int(W/g)
    for x in range(W):
        for y in range(H):
            for z in range(C):
                out[x*g :(x+1)*g, y*g : (y+1)*g, z] = np.max(out[x*g :(x+1)*g, y*g : (y+1)*g, z]).astype(np.int)
    return out

img = cv2.imread("imori.jpg")

out = mean_pool(img)

cv2.imwrite("out_figure8.jpg",out)


