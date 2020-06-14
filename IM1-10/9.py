import cv2
import numpy as np

def gausu(img):
    filter = 3
    pad = filter // 2
    H,W,C = img.shape
    img1 = np.zeros((H+pad*2,W+pad*2,C),dtype = np.float)
    img1[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)
    out = np.zeros_like(img, dtype = np.float)

    K = [[1., 2., 1.],
    [2., 4., 2.],
    [1., 2., 1.]]

    for y in range(H):
        for x in range(W):
            for z in range(C):
                out[y,x,z] = np.sum(K * (img1[y:filter+y, x:filter+x, z])/16)
    
    return out

img1 = cv2.imread("imori_noise.jpg")

out = gausu(img1)

cv2.imwrite("out_figure9.jpg",out)

