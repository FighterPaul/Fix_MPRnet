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



print(f"{'*'*10}  Start {__file__}")


#------------------------------------ Parser ------------------------------------

parser = argparse.ArgumentParser(description='Demo MPRNet')
parser.add_argument('--input_dir', default='./samples/input/', type=str, help='Input images')
parser.add_argument('--result_dir', default='./samples/output/', type=str, help='Directory for results')
parser.add_argument('--task', required=False, default='Deraining', type=str, help='Task to run', choices=['Deblurring', 'Denoising', 'Deraining'])
args = parser.parse_args()

task    = args.task
input_dir = args.input_dir
result_dir = args.result_dir

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

def prune_model_YOLO(model, amount=0.2):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module=module, name='weight')

    return model


def prune_model_MPRNet(model, amount=0.2):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module=module, name='weight')

    return model




#---------------------- Check system -----------------------------------------
print(f"cuda available :: {torch.cuda.is_available()}")
os.makedirs(result_dir, exist_ok=True)


BBOX_COLOR = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]





#----------------------  Initial Model YOLO -------------------------------------
print("Initilize YOLO ....")
YOLO_MODEL = YOLO(model='./yolo11s.pt', task='detect')
YOLO_MODEL.cuda()
LABELS = YOLO_MODEL.names



print("Pruning YOLO Model ...")
yolo_model_prepare_prune = YOLO_MODEL.model
yolo_model_after_prune = prune_model_YOLO(model=yolo_model_prepare_prune, amount=0.1)
print("YOLO Model pruned")
YOLO_MODEL.model = yolo_model_after_prune
print("saving pruned YOLO model ....")
YOLO_MODEL.save('yolo11s_trained_pruned.pt')
print("Pruned YOLO Model saved.")

print("Initilize YOLO ....")
YOLO_MODEL_PRUNED = YOLO(model='yolo11s_trained_pruned.pt')
YOLO_MODEL_PRUNED.cuda()





#--------------------   Initial Model MPRNet --------------------------------------
print("Initilize MPRNet ....")
load_file = run_path(os.path.join("MPRNet.py"))
MPRNet_MODEL = load_file['MPRNet']()
MPRNet_MODEL.cuda()
MPRNet_MODEL_WEIGHT = './model_deraining.pth'
load_checkpoint(MPRNet_MODEL, MPRNet_MODEL_WEIGHT)


# print("Pruning MPRNet Model ...")
# mprnet_model_prepare_prune = MPRNet_MODEL
# mprnet_model_after_prune = prune_model_MPRNet(model=mprnet_model_prepare_prune, amount=0.1)
# print("MPRNet Model pruned")
# MPRNet_MODEL = mprnet_model_after_prune
# print("saving pruned MPRNet model ....")
# torch.save(obj= MPRNet_MODEL.state_dict(),f='./MPRNet_trained_pruned.pth')
# print("Pruned MPRNet Model saved.")



# print("Initilize MPRNet ....")
# load_file = run_path(os.path.join("MPRNet.py"))
# MPRNet_MODEL_PRUNED = load_file['MPRNet']()
# MPRNet_MODEL_PRUNED.cuda()
# MPRNet_MODEL_WEIGHT_PRUNED = './MPRNet_trained_pruned.pth'
# load_checkpoint(MPRNet_MODEL_PRUNED, MPRNet_MODEL_WEIGHT_PRUNED)
















#-----------------------   Initial USB Camera ------------------------------
# SOURCE_TYPE = 'usb'
# USB_IDX = 0
# RES_W, RES_H = 480, 320


# cap_arg = USB_IDX
# cap = cv2.VideoCapture(cap_arg)
# cap.set(3, RES_W)
# cap.set(4, RES_H)


#----------------------  edit dimension of image -----------------------
print("*** PHASE edit images dimension ***")
images = natsorted(glob(os.path.join(input_dir, '*.jpg'))
                + glob(os.path.join(input_dir, '*.JPG'))
                + glob(os.path.join(input_dir, '*.png'))
                + glob(os.path.join(input_dir, '*.PNG')))

INTERMEDITE_FOLDER = './Images/EditDimension_images'
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



images = natsorted(glob(os.path.join(INTERMEDITE_FOLDER, '*.jpg'))
                + glob(os.path.join(INTERMEDITE_FOLDER, '*.JPG'))
                + glob(os.path.join(INTERMEDITE_FOLDER, '*.png'))
                + glob(os.path.join(INTERMEDITE_FOLDER, '*.PNG')))

#---------------------- image inference ---------------------------------
with torch.no_grad():
    for loop_idx, each_image in enumerate(images):

        print(f"Loop {loop_idx}")
    
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
        img = PIL.Image.open(each_image).convert('RGB')
        input_ = TF.to_tensor(img).unsqueeze(0).cuda()

        restored_image = MPRNet_MODEL(input_)
        restored_image = restored_image[0]
        restored_image = torch.clamp(restored_image, 0, 1)

        restored_image = restored_image.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored_image = img_as_ubyte(restored_image[0])

        restored_image = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)




#--------------------- YOLO Predict -----------------------------
        detected_image = YOLO_MODEL_PRUNED(source=restored_image)
        detection_results = detected_image[0].boxes
        print(f"detect image {loop_idx}")

        object_count = 0
        for i in range(len(detection_results)):
            box_tensor = detection_results[i].xyxy.cpu()
            xyxy = box_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            classidx = int(detection_results[i].cls.item())
            classname = LABELS[classidx]
            conf = detection_results[i].conf.item()

            if conf > 0.5:
                color_rectangle = BBOX_COLOR[classidx % 10]
                cv2.rectangle(restored_image, (xmin,ymin), (xmax,ymax), color=color_rectangle, thickness=2)
                label = f'{classname} : {int(conf * 100)} %'
                labelSize, baseline = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(restored_image, pt1=(xmin, label_ymin - labelSize[1] - 10), pt2=(xmin + labelSize[0], label_ymin + baseline - 10), color=color_rectangle, thickness= cv2.FILLED)
                cv2.putText(img=restored_image, text=label, org=(xmin, label_ymin - 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=1)

                object_count += 1



#-------------------- save image ----------------------
        path_to_save = os.path.join(result_dir, os.path.basename(each_image))
        cv2.imwrite(filename=path_to_save, img=restored_image)