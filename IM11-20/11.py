import cv2
import numpy as np

def mean_filter(img):
    filter = 3
    pad = filter // 2
    H,W,C = img.shape
    img1 = np.zeros((H+pad*2,W+pad*2,C),dtype = np.float)
    img1[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)
    out = np.zeros_like(img, dtype = np.uint)
    for x in range(H):
        for y in range(W):
            for z in range(C):
                img1[x+pad, y+pad] = img[x,y]
    for x in range(H):
        for y in range(W):
            for z in range(C):
                out[x, y, z] = np.mean(img1[x:x+filter, y:y+filter, z]).astype(np.uint8)
    return out

img = cv2.imread("imori.jpg")

out1 = mean_filter(img)

cv2.imwrite("out_figure11.jpg",out1)
