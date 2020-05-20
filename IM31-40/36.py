import cv2
import numpy as np

T, K = 8, 8
channel = 3

def w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return (( 2 * cu * cv / T) * np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

def dct(img):
    H,W,_ = img.shape
    out = np.zeros_like(img,dtype = np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for v in range(T):
                    for u in range(T):
                        for y in range(T):
                            for x in range(T):
                                out[v+yi, u+xi, c] += img[y+yi, x+xi,c] * w(x,y,u,v)
    return out

def idct(img):
    out
    return out

img = cv2.imread("imori.jpg")

img1 = dct(img)
out = idct(img1)

cv2.imwrite(out, "out.jpg")
