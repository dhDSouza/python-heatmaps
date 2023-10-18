import cv2
import numpy as numpy
from ultralytics import YOLO

video = cv2.VideoCapture('data/carros.mp4')

while True:
    check, img = video.read()
    img = cv2.resize(img, (1270, 720))

    cv2.imshow('Img', img)
    cv2.waitKey(1)