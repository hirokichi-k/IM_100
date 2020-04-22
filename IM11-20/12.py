import cv2
import numpy as np

def motion_filter(img):
    filter_size = 3
    K = np.diag([1]*filter_size).astype(np.float)
    K /= filter_size

    pad = filter_size // 2
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
                out[x, y, z] = np.sum(K * img1[x:x+filter_size, y:y+filter_size, z]).astype(np.uint8)
    return out

img = cv2.imread("imori.jpg")

out1 = motion_filter(img)

cv2.imwrite("out_figure12.jpg",out1)
