import cv2
import numpy as np

def median_filter(img):
    filter = 3
    pad = filter // 2
    H,W,C = img.shape
    img1 = np.zeros((H+pad*2,W+pad*2,C),dtype = np.float)
    img1[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)
    out = np.zeros_like(img, dtype = np.float)
    for x in range(H):
        for y in range(W):
            for z in range(C):
                img1[x+pad, y+pad] = img[x,y]
    for x in range(H):
        for y in range(W):
            for z in range(C):
                out[x, y, z] = np.median(img1[x:x+filter,y:y+filter,z])
    return out

img = cv2.imread("imori_noise.jpg")

out = median_filter(img)



cv2.imwrite("out_figure10.jpg",out)
