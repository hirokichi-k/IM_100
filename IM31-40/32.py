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
                G[l, k ,c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x*k/W + y*l/H)))/np.sqrt(W*H)
                return G


# def idft(G):
#     H, W, C = G.shape
#     out = np.zeros(G.shape, dtype = np.float32)

#     x = np.tile(np.arange(W), (H, 1))
#     y = np.arange(H).repeat(W).reshape(H,-1)

#     for c in range(C):
#         for l in range(H):
#             for k in range(W):
#                 out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

#     out = np.clip(out, 0, 255).astype(np.uint8)
#     return out


# def gray(img):
#     img[:, :, 0] *= 0.0722 #B
#     img[:, :, 1] *= 0.7152 #G
#     img[:, :, 2] *= 0.2126 #R
#     img2 = np.sum(img, axis=2).astype(np.uint8)
#     return img2

# DFT hyper-parameters
K, L = 128, 128
channel = 3



# IDFT
def idft(G):
    # prepare out image
    H, W, _ = G.shape
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



img = cv2.imread("imori.jpg").astype(np.float32)

G = dft(img)

ps = (np.abs(G) / np.abs(G).max() * 255).astype(np.uint8)
cv2.imwrite("out_ps.jpg", ps)

out = idft(G)

cv2.imwrite("out.jpg", out)

# gray = gray(img)

# fimg = np.fft.fft2(gray)
    
# # 第1象限と第3象限, 第2象限と第4象限を入れ替え
# fimg =  np.fft.fftshift(fimg)
# print(fimg.shape)
# # パワースペクトルの計算
# mag = 20*np.log(np.abs(fimg))
    
# # 入力画像とスペクトル画像をグラフ描画
# plt.subplot(121)
# plt.imshow(gray, cmap = 'gray')
# plt.subplot(122)
# plt.imshow(mag, cmap = 'gray')
# plt.show()




