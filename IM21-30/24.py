import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def gamma_corre(img, c = 1, g = 2.2):
    out = img.copy().astype(np.float)

    out /= 255.
    out = (out/c)**(1/g)
    out *= 255

    out = out.astype(np.uint8)

    return out


img = cv2.imread("imori_gamma.jpg").astype(np.float)

out = gamma_corre(img,c = 1, g = 2.2)

plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out_gamma.png")

cv2.imwrite("out_gamman.jpg",out)

