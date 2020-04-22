import cv2
import numpy as np

def mean_filter(img):
    filter = 3
    pad = filter // 2
    H,W,C = img.shape
    img1 = np.zeros((H+pad*2,W+pad*2,C),dtype = np.float)
    img1[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)
    out = np.zeros_like(img, dtype = np.uint)
    out1 = np.zeros_like(img1, dtype = np.uint)
    out2 = np.zeros_like(img1, dtype = np.uint)

    for x in range(H):
        for y in range(W):
            for z in range(C):
                img1[x+pad, y+pad] = img[x,y]

    for x in range(H):
        for y in range(W):
            for z in range(C):
                out1[x, y, z] = np.max(img1[x:x+filter, y:y+filter, z]).astype(np.uint8)
                out2[x, y, z] = np.min(img1[x:x+filter, y:y+filter, z]).astype(np.uint8)
    out = out1 - out2
    return out

img = cv2.imread("imori.jpg")

out1 = mean_filter(img)

cv2.imwrite("out_figure13.jpg",out1)
