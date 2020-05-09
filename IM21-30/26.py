import cv2
import numpy as np

def bi_linear(img, ax = 1.5, ay = 1.5):
    W,H,_ = img.shape
    aH = int(H*ay)
    aW = int(W*ax)

    y = np.arange(aH).repeat(aW).reshape(aH,-1)/ay
    x = np.tile(np.arange(aW), (aH,1))/ax

    iy = np.floor(y).astype(np.int)
    ix = np.floor(x).astype(np.int)

    ix = np.minimum(ix, W-2)
    iy = np.minimum(iy, H-2)

    dy = y - iy
    dx = x - ix

    dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
    dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)

    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]
    
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    return out


img = cv2.imread("imori.jpg").astype(np.float)

out = bi_linear(img, ax = 1.5, ay = 1.5)

cv2.imwrite("out_big2.jpg",out)

