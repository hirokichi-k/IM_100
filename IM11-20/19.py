import cv2
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def gray(img):
<<<<<<< HEAD
    out = img[:,:,0] * 0.0722 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.2126
=======
    img[:,:,0] *= 0.0722
    img[:,:,1] *= 0.7152
    img[:,:,2] *= 0.2126
    out = np.sum(img)
>>>>>>> 4f304831c12b5e80dedba17f836006fca4753ab0
    out = out.astype(np.uint8)
    return out

def LoG(img, K_size = 5, sigma = 3):
    H,W = img.shape
    
    pad = K_size//2
<<<<<<< HEAD
    out = np.zeros((H + pad*2 ,W + pad*2), dtype = np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = (x ** 2 + y ** 2 - 2 * (sigma ** 2)) * np.exp( - (x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * (sigma ** 6))
    K /= K.sum()

    # print(K)

    for y in range(H):
        for x in range(W):xx
            out[pad + y, pad + x] = np.sum(K * out[y:y+K_size,x:x+K_size])
    
    out = np.clip(out, 0, 255)
    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

=======
    
    out = img
>>>>>>> 4f304831c12b5e80dedba17f836006fca4753ab0
    return out

img = cv2.imread("imori_noise.jpg").astype(np.float)

img1 = gray(img)
img2 = LoG(img1, K_size = 5, sigma = 3)

cv2.imwrite("out_figure19.jpg",img2)



