import numpy as np
import cv2

def affine_prallel(img, tx, ty):
    H, W, C = img.shape
    img1 = np.zeros((H+2,W+2,C),dtype=np.float)
    img1[1:H+1,1:W+1] = img

    out = np.zeros((H, W, C), dtype=np.float32)

    x_new = np.tile(np.arange(W),(H,1))
    y_new = np.arange(H).repeat(W).reshape(H,-1)

    x = x_new - tx
    y = y_new - ty

    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

    out[y_new, x_new] = img1[y,x]
    out = out[:H, :W]
    out = out.astype(np.uint8)

    return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Affine
out = affine_prallel(img, tx=30, ty=-30)
# print(out.shape)
# Save result
cv2.imwrite("out.jpg", out)




