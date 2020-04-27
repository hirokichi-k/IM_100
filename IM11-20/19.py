import cv2
import numpy as np

def gray(img):
    img[:,:,0] *= 0.0722
    img[:,:,1] *= 0.7152
    img[:,:,2] *= 0.2126
    out = np.sum(img)
    out = out.astype(np.uint8)
    return out

def LoG(img, K_size = 5, sigma = 3):
    H,W = img.shape
    
    pad = K_size//2
    
    out = img
    return out

img = cv2.imread("imori_noise.jpg")

img1 = gray(img)
img2 = LoG(img1, K_size = 5, sigma = 3)

cv2.imwrite("out_figure19.jpg",img2)

