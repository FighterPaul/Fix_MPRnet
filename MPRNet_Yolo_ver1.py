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



print(f"{'*'*10}  Start {__file__}")


#------------------------------------ Parser ------------------------------------

parser = argparse.ArgumentParser(description='Demo MPRNet')
parser.add_argument('--input_dir', default='./samples/input/', type=str, help='Input images')
parser.add_argument('--result_dir', default='./samples/output/', type=str, help='Directory for results')
parser.add_argument('--task', required=False, default='Deraining', type=str, help='Task to run', choices=['Deblurring', 'Denoising', 'Deraining'])
args = parser.parse_args()

task    = args.task
inp_dir = args.input_dir
out_dir = args.result_dir

#-------------------------------------------------------------------------------------

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)











#----------------------  Initial Model YOLO -------------------------------------
# YOLO_MODEL = YOLO(model='./yolo11s.pt', task='detect')
# labels = YOLO_MODEL.names


#--------------------   Initial Model MPRNet --------------------------------------
# load_file = run_path(os.path.join("MPRNet.py"))
# MPRNet_MODEL = load_file['MPRNet']()
# MPRNet_MODEL.cuda()
# MPRNet_MODEL_WEIGHT = './model_deraining.pth'
# load_checkpoint(MPRNet_MODEL, MPRNet_MODEL_WEIGHT)







#-----------------------   Initial USB Camera ------------------------------
# SOURCE_TYPE = 'usb'
# USB_IDX = 0
# RES_W, RES_H = 480, 320


# cap_arg = USB_IDX
# cap = cv2.VideoCapture(cap_arg)
# cap.set(3, RES_W)
# cap.set(4, RES_H)


images = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                + glob(os.path.join(inp_dir, '*.JPG'))
                + glob(os.path.join(inp_dir, '*.png'))
                + glob(os.path.join(inp_dir, '*.PNG')))

INTERMEDITE_FOLDER = './Images/EditDimension_images'
#----------------------  edit dimension of image -----------------------
for loop_idx, each_image in enumerate(images):
    print(f"loop IDX :: {loop_idx}")
    im = cv2.imread(each_image, cv2.IMREAD_COLOR)
    if im.shape[:2] != (480, 320):
        print("found image wrong shape ::", im.shape[:2])
        im = cv2.resize(src=im, dsize=(480, 320), dst=None, fx= None, interpolation=cv2.INTER_LINEAR)
    else:
        print("image good shape", im.shape[:2])

    path_to_save_overwirte = os.path.join(INTERMEDITE_FOLDER, os.path.basename(each_image))
    print(f"saving at {path_to_save_overwirte}")
    cv2.imwrite(filename=path_to_save_overwirte, img=im)

#-------------------------------------------------------------------------------



#---------------------- image ---------------------------------
for loop_idx, each_image in enumerate(images):
    pass
    
    # ret, frame = cap.read()
    # if (frame is None) or (not ret):
    #     print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
    #     break

    # frame = cv2.resize(frame(RES_W, RES_H))
    
    # cv2.imshow(winname= 'USB Camera 00', mat=frame)
    # key = cv2.waitKey(5)

    # if key ==('q') or key == ord('Q'):
    #     break












#---------------------  Restoration -----------------------------






#--------------------- YOLO Predict -----------------------------