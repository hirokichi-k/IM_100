import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

img = cv2.imread("imori_dark.jpg").astype(np.float)

plt.hist(img.ravel(), bins = 255, rwidth=0.8, range = (0,255))
plt.savefig("out.png")
plt.show()

