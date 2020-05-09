import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def hist_norm(img):
    a,b = 0,255
    c = np.min(img)
    d = np.max(img)

    out = img.copy()
    out  = (b-a) * (out - c) / (d-c) + a

    print(c,d)
    return out


img = cv2.imread("imori_dark.jpg").astype(np.float)

out = hist_norm(img)

plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out_his.png")

cv2.imwrite("out_hist.jpg",out)

