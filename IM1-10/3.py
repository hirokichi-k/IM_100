import cv2
import numpy as np
img = cv2.imread("imori.jpg").astype(np.float32)


# print("%d", img[:, :, 0])

img[:, :, 0] *= 0.0722 #B
img[:, :, 1] *= 0.7152 #G
img[:, :, 2] *= 0.2126 #R

# img2 = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]

img2 = np.sum(img, axis=2).astype(np.uint8)

for(i = 0 ; i < img2[][].length ; i++)
