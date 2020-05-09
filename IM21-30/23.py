import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def hist_equation(img, Zmax = 255):
    H,W,C = img.shape
    S = H*W*C*1.

    out = img.copy()
    h_pool = 0.

    for i in range(1, Zmax):
        h_pool += len(out[out == i])
        out[out == i] = Zmax/S * h_pool

    out = out.astype(np.uint8)

    return out


img = cv2.imread("imori.jpg").astype(np.float)

out = hist_equation(img, Zmax = 255)

plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out_equation.png")

cv2.imwrite("out_equation.jpg",out)

