import cv2
import numpy as np

def gray(img):
    b=img[:,:,0]
    g=img[:,:,1]
    r=img[:,:,2]
    out = np.sum(0.2126 * r, 0.7152 * g, 0.0722 * b)
    out = out.astype(np.nint8)
    return out

def LoG(img):
    out = img
    return out

img = cv2.imread("imori_noise.jpg")

img1 = gray(img)
img2 = LoG(img1)

cv2.imwrite("out_figure19.jpg",img2)

