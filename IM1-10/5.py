import cv2
import numpy as np



def RGB2HSV(_img):
    img = _img.copy()/255.
    hsv = np.zeros_like(img, dtype = np.float32)
    max = np.max(img, axis = 2)
    min = np.min(img, axis = 2)
    argmin = np.argmin(img, axis = 2)

    t = np.where((argmin == 0) & (max != min)) #min = B
    hsv[...,0][t] = 60*(img[...,1][t] - img[...,2][t]) / (max[t] - min[t]) + 60

    t = np.where(argmin == 2) #min = R
    hsv[...,0][t] = 60*(img[...,0][t] - img[...,1][t]) / (max[t] - min[t]) + 180

    t = np.where(argmin == 1) #min = G
    hsv[...,0][t] = 60*(img[...,2][t] - img[...,0][t]) / (max[t] - min[t]) + 300

    t = np.where(max == min)
    # print(t)
    hsv[...,0][t] = 0
    
    hsv[...,1] = max - min #S
    hsv[...,2] = max #V
    return hsv

def HSV2RGB(_img, hsv):
    img = _img.copy()/255
    max = np.max(img, axis=2)
    min = np.min(img, axis=2)

    out = np.zeros_like(img)

    C = hsv[...,1]
    H = hsv[...,0]/60
    V = hsv[...,2]
    X = C * (1 - np.abs(H % 2 -1))
    Z = np.zeros_like(H)

    vals = [[C,X,Z],[X,C,Z],[Z,C,X],[Z,X,C],[X,Z,C],[C,Z,X]]


    for i in range(6):
        t = np.where((i <= H) & (H < i+1))
        out[...,2][t] = vals[i][0][t] + (V - C)[t]
        out[...,1][t] = vals[i][1][t] + (V - C)[t]
        out[...,0][t] = vals[i][2][t] + (V - C)[t]

    out[max == min] = 0
    out = np.clip(out,0,1)
    out = (out*255).astype(np.uint8)

    return out


img = cv2.imread('imori.jpg').astype(np.float32)

hsv = RGB2HSV(img)

hsv[...,0] = (hsv[...,0] + 180) % 360

out = HSV2RGB(img,hsv)

cv2.imwrite("out_figure5.jpg",out)

