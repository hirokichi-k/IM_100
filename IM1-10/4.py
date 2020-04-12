import cv2
import numpy as np

def gray(img):
    img[:, :, 0] *= 0.0722 #B
    img[:, :, 1] *= 0.7152 #G
    img[:, :, 2] *= 0.2126 #R
    img2 = np.sum(img, axis=2).astype(np.uint8)
    return img2

def oo2tika(img):
    max_sigma = 0
    max_i = 0
    H,W = img.shape
    for i in range(1, 256):
        v0 = img[np.where(img < i)]
        w0 = len(v0)/(H*W)
        m0 = np.mean(v0) if w0 > 0 else 0.
        
        v1 = img[np.where(img >= i)] 
        w1 = len(v1)/(H*W)
        m1 = np.mean(v1) if w1 > 0 else 0.
        
        sigma = w0 * w1 * ((m0 - m1)**2)

        if sigma > max_sigma:
            max_i = i
            max_sigma = sigma

    return max_i


img = cv2.imread('imori.jpg').astype(np.float32)
img2 = gray(img)

th = oo2tika(img2)
print(th)

img3 = np.where(img2 < th, 0, 255)
cv2.imwrite("out_figure4.jpg",img3)
