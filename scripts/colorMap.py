import cv2

img = cv2.imread('data/img02.png')

if img is not None:
    imgColor = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    cv2.imshow('Img', img)
    cv2.imshow('ColorMap', imgColor)

    cv2.waitKey(0)
else:
    print("Imagem não encontrada.")
