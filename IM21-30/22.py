import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def hist_change(img, m0 = 128, s0 = 52):
    m = np.mean(img)
    s = np.std(img)

    out = img.copy()
    out  = s0 * (out - m) / s + m0

    return out


img = cv2.imread("imori_dark.jpg").astype(np.float)

out = hist_change(img, m0 = 128, s0 = 52)

plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out_change.png")

cv2.imwrite("out_change.jpg",out)


