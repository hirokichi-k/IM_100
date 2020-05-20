
import cv2
import numpy as np
import matplotlib.pyplot as plt


def dft(img):
    H,W,C = img.shape

    G = np.zeros(img.shape, dtype = np.complex)

    x = np.tile(np.arange(W),(H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)

    for c in range(C):
        for l in range(H):
            for k in range(W):
                G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x*k/W + y*l/H)))/np.sqrt(W*H)
                return G

def idft(G):
    # prepare out image
    H, W, channel = G.shape
    out = np.zeros((H, W, channel), dtype=np.float32)

    # prepare processed index corresponding to original image positions
    x = np.tile(np.arange(W), (H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)

    # idft
    for c in range(channel):
        for l in range(H):
            for k in range(W):
                out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

    # clipping
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

def lpf(G):
    ratio = 0.5
    H, W, channel = G.shape	

    # transfer positions
    _G = np.zeros_like(G)
    _G[:H//2, :W//2] = G[H//2:, W//2:]
    _G[:H//2, W//2:] = G[H//2:, :W//2]
    _G[H//2:, :W//2] = G[:H//2, W//2:]
    _G[H//2:, W//2:] = G[:H//2, :W//2]

    # get distance from center (H / 2, W / 2)
    x = np.tile(np.arange(W), (H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)

    # make filter
    _x = x - W // 2
    _y = y - H // 2
    r = np.sqrt(_x ** 2 + _y ** 2)
    mask = np.ones((H, W), dtype=np.float32)
    mask[r > (W // 2 * ratio)] = 0

    mask = np.repeat(mask, channel).reshape(H, W, channel)

    # filtering
    _G *= mask

    # reverse original positions
    G[:H//2, :W//2] = _G[H//2:, W//2:]
    G[:H//2, W//2:] = _G[H//2:, :W//2]
    G[H//2:, :W//2] = _G[:H//2, W//2:]
    G[H//2:, W//2:] = _G[:H//2, :W//2]
    return G


img = cv2.imread("imori.jpg").astype(np.float32)

G = dft(img)

G = lpf(G)

out = idft(G)

cv2.imwrite("out.jpg", out)

