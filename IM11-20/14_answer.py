import cv2
import numpy as np

# Gray scale
def BGR2GRAY(img):
	b = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	r = img[:, :, 2].copy()

	# Gray scale
	out = 0.2126 * r + 0.7152 * g + 0.0722 * b
	out = out.astype(np.uint8)

	return out

# different filter
def different_filter(img, K_size=3):
	if len(img.shape) == 3:
		H, W, C = img.shape
	else:
		img1 = img.copy()
		img = np.expand_dims(img, axis=-1)
		H, W, C = img.shape

	# Zero padding
	pad = K_size // 2
	out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
	out[pad: pad + H, pad: pad + W] = img1.copy().astype(np.float)
	tmp = out.copy()

	out_v = out.copy()
	out_h = out.copy()

	# vertical kernel
	Kv = [[0., -1., 0.],[0., 1., 0.],[0., 0., 0.]]
	# horizontal kernel
	Kh = [[0., 0., 0.],[-1., 1., 0.], [0., 0., 0.]]

	# filtering
	for y in range(H):
		for x in range(W):
			out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
			out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y: y + K_size, x: x + K_size]))

	out_v = np.clip(out_v, 0, 255)
	out_h = np.clip(out_h, 0, 255)

	out_v = out_v[pad: pad + H, pad: pad + W].astype(np.uint8)
	out_h = out_h[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out_v, out_h

# Read image
img = cv2.imread("imori.jpg").astype(np.float)

# grayscale
gray = BGR2GRAY(img)

# different filtering
out_v, out_h = different_filter(gray, K_size=3)




cv2.imwrite("out_figure14_tate.jpg",out_v)
cv2.imwrite("out_figure14_yoko.jpg",out_h)