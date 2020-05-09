import numpy as np
import cv2

def affine(img, tx, ty, a, b, c, d):
    H, W, C = img.shape
    img1 = np.zeros((H+2,W+2,C),dtype=np.float)
    img1[1:H+1,1:W+1] = img

    aH = int(H*d)
    aW = int(W*a)
    print(aH,aW)

    out = np.zeros((aH, aW, C), dtype=np.float32)

    x_new = np.tile(np.arange(aW),(aH,1))
    y_new = np.arange(aH).repeat(aW).reshape(aH,-1)

    adbc = a * d - b * c
    x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

    # x = np.clip(x, 0, W+1)
    # y = np.clip(y, 0, W+1)

    out[y_new, x_new] = img1[y,x]
    out = out[:aH, :aW]
    out = out.astype(np.uint8)

    return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Affine
out = affine(img, tx=30, ty=-30, a=1.3, b=0, c=0, d=0.8)
print(out.shape)
# Save result
cv2.imwrite("out.jpg", out)




