import cv2
import numpy as np

def gray(img):
    # Gray scale
    out = 0.2126 * img[:, :, 2].copy() + 0.7152 * img[:, :, 1].copy() + 0.0722 *  img[:, :, 0].copy()
    out = out.astype(np.uint8)
    return out

def motion_filter(img):
    filter_size = 3

    K1 = [[0., 0., 0.],[(-1.), 1., 0.], [0., 0., 0.]]

    K = [[0., -1., 0.],
    [0., 1., 0.],
    [0., 0., 0.]]


    pad = filter_size // 2
    # img_gray = img.copy()
    # img = np.expand_dims(img, axis=-1)
    H,W = img.shape
    
    img1 = np.zeros((H+pad*2,W+pad*2),dtype = np.float)
    img1[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)
    out = np.zeros_like(img1)
    out1 = np.zeros_like(img1)

    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(K * (img1[y:y+filter_size, x:x+filter_size]))
            out1[y, x] = np.sum(K1 * (img1[y:y+filter_size, x:x+filter_size]))
    out = np.clip(out, 0, 255).astype(np.uint8)
    out1 = np.clip(out1, 0, 255).astype(np.uint8)

    # for y in range(H):
    #     for x in range(W):
    #         out[pad + y, pad + x] = np.sum(K * (img1[y: y + filter_size, x: x + filter_size]))
    #         out1[pad + y, pad + x] = np.sum(K1 * (img1[y: y + filter_size, x: x + filter_size]))

    # out = np.clip(out, 0, 255)
    # out1 = np.clip(out1, 0, 255)

    # out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    # out1 = out1[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out, out1

img = cv2.imread("imori.jpg").astype(np.float)

img1 = gray(img)

out,out1 = motion_filter(img1)


cv2.imwrite("out_figure14_tate.jpg",out)
cv2.imwrite("out_figure14_yoko.jpg",out1)

