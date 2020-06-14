import cv2
import numpy as np

T, K = 8, 4
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
    out = np.clip(np.round(out), 0, 255)
    return out

def idct(img):
    H, W, _ = img.shape
    out = np.zeros_like(img, dtype= np.float32)
    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for y in range(T):
                    for x in range(T):
                        for v in range(K):
                            for u in range(K):
                                out[y+yi, x+xi, c] += img[v+yi, u+xi, c] * w(x, y, u, v)
    out = np.clip(np.round(out).astype(np.uint8), 0, 255)
    return out

def quantization(F):
    H, W, _ = F.shape

    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                (12, 12, 14, 19, 26, 58, 60, 55),
                (14, 13, 16, 24, 40, 57, 69, 56),
                (14, 17, 22, 29, 51, 87, 80, 62),
                (18, 22, 37, 56, 68, 109, 103, 77),
                (24, 35, 55, 64, 81, 104, 113, 92),
                (49, 64, 78, 87, 103, 121, 120, 101),
                (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
    for ys in range(0, H, T):
        for xs in range(0, W, T):
            for c in range(channel):
                F[ys: ys + T, xs: xs + T, c] =  np.round(F[ys: ys + T, xs: xs + T, c] / Q) * Q
    return F

def MSE(img1, out):
    H,W,_ = img1.shape
    return np.sum((img1 - out)**2)/(H*W*channel)

def PSNR(mse):
    vmax = 255
    return 10*np.log10(vmax*vmax/mse)

def bitrate():
    return 1. * T * K * K /(T*T)


img = cv2.imread("imori.jpg").astype(np.float32)

img1 = dct(img).astype(np.float32)
out = idct(img1)

mse = MSE(img, out)

psnr = PSNR(mse)

bit = bitrate()

print("MSE:", mse)
print("PSNR:", psnr)
print("bitrate:", bit)
