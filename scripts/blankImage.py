import cv2
import numpy as np

h = 720
w = 1270

img = np.ones([h, w], dtype=np.uint8)

for x in range(256):
    img[100:200, 100:200] = x
    img[500:600, 500:600] = x
    img[100:200, 900:1000] = x

    heatMap = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    cv2.imshow('Img', heatMap)
    cv2.waitKey(30)
