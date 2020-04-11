import cv2
img = cv2.imread("imori.jpg")
blue = img[:, :, 0].copy()
green = img[:, :, 1].copy()
red = img[:, :, 2].copy()

img[:, :, 0] = red
img[:, :, 2] = blue

cv2.imwrite("img_out.jpg", img)
