import cv2
import numpy as np

def gray(img1):
    out = 0.2126 * img1[..., 2] + 0.7152 * img1[..., 1] + 0.0722 * img1[..., 0]
    return out

def oo2tika(img2):
    max_i = 0
    max_sigma = 0
    for i in range(1,255):
        class0 = img2[np.where(img2 < i)]
        class1 = img2[np.where(img2 >= i)]
        w0 = len(class0)
        w1 = len(class1)
        m0 = np.mean(class0)
        m1 = np.mean(class1)
        sigma = w0 * w1 * ((m0 - m1)**2) / ((w0 + w1)**2)
        if max_sigma < sigma:
            max_sigma = sigma
            max_i = i
    return max_i


img = cv2.imread("imori.jpg").astype(np.float32)

out_gray = gray(img).astype(np.uint8)

i = oo2tika(out_gray)

out_2tika = np.where(out_gray < i, 0, 255)

cv2.imwrite("ex4.jpg",out_2tika)

