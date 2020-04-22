import cv2
import numpy as np

def gausu(img):
    filter = 3
    pad = filter // 2
    H,W,C = img.shape
    img1 = np.zeros((H+pad*2,W+pad*2,C),dtype = np.float)
    img1[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)
    out = np.zeros_like(img, dtype = np.float)
    
    return out

img1 = cv2.imread("imori_noise.jpg")

out = gausu(img1)

cv2.imwrite("out_figure9.jpg",out)

