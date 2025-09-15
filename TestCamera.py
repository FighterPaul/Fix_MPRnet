import PIL.Image
import torch.nn.utils.prune as prune
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import PIL
import os
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
import time
from torchinfo import summary



SOURCE_TYPE = 'usb'
USB_IDX = 0
RES_W, RES_H = 640, 480


cap_arg = USB_IDX
# cap = cv2.VideoCapture(cap_arg)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FPS, 30)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# cap.set(3, RES_W)
# cap.set(4, RES_H)


cap = cv2.VideoCapture(USB_IDX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

for _ in range(15):
    cap.read()


while True:
    ret, frame = cap.read()
    if (frame is None) or (not ret):
        print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
        break

    frame = cv2.resize(frame, (RES_W, RES_H))

    cv2.imshow(winname= 'USB Camera 00', mat=frame)
    key = cv2.waitKey(5)

    if key ==('q') or key == ord('Q'):
        break