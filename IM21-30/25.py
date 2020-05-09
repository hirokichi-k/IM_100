import cv2
import numpy as np

def nn_interpolate(img, ax = 1.5, ay = 1.5):
    W,H,_ = img.shape
    aH = int(H*ay)
    aW = int(W*ax)

    y = np.arange(aH).repeat(aW).reshape(aH,-1)
    x = np.tile(np.arange(aW), (aH,1))
    y = np.round(y/ay).astype(np.int)
    x = np.round(x/ax).astype(np.int)

    out = img[y,x]
    out = out.astype(np.uint8)
    return out


img = cv2.imread("imori.jpg").astype(np.float)

out = nn_interpolate(img, ax = 1.5, ay = 1.5)

cv2.imwrite("out_big.jpg",out)

