import numpy as np
import cv2

def affine(img, dx, dy):
    H, W, C = img.shape
    a, b, c, d, tx, ty = [1., dx/H, dy/W, 1., 0., 0.]
    img1 = np.zeros((H+2,W+2,C),dtype=np.float32)
    img1[1:H+1,1:W+1] = img

    aH = np.ceil(H+dx).astype(np.int)
    aW = np.ceil(W+dy).astype(np.int)
    # print(aH,aW)

    out = np.zeros((aH, aW, C), dtype=np.float32)

    x_new = np.tile(np.arange(aW),(aH,1))
    y_new = np.arange(aH).repeat(aW).reshape(aH,-1)

    adbc = a * d - b * c
    x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

    # dcx = (x.max() + x.min()) // 2 - W // 2
    # dcy = (y.max() + y.min()) // 2 - H // 2

    # x -= dcx
    # y -= dcy

    x = np.clip(x, 0, W + 1).astype(np.int)
    y = np.clip(y, 0, H + 1).astype(np.int)
    
    out[y_new, x_new] = img1[y,x]
    out = out.astype(np.uint8)

    return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)


# Affine
out = affine(img, dx=30, dy=30)
print(out.shape)
# Save result
cv2.imwrite("out.jpg", out)




