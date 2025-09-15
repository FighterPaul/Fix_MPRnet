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
from torch.ao.quantization import quantize_dynamic
# from torchsummary import summary



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

#---------------------------------------- Utilis Function ---------------------------------------------

def load_checkpoint(model, weights_path):
    checkpoint = torch.load(f= weights_path, weights_only= True)
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
        if isinstance(module, torch.nn.Sequential):
            # print("This module is Sequential", end='  ')
            for sub_module in module:
                if isinstance(sub_module, torch.nn.Linear):
                    # print(" sub module is Linear ")
                    prune.l1_unstructured(sub_module, name='weight', amount=amount)
                    prune.remove(module=sub_module, name='weight')
                elif isinstance(sub_module, torch.nn.Conv2d):
                    # print(" sub module is Conv ")
                    prune.l1_unstructured(sub_module, name='weight', amount=amount)
                    prune.remove(module=sub_module, name='weight')


        elif isinstance(module, torch.nn.Linear):
            # print("This module is Linear")
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module=module, name='weight')
        elif isinstance(module, torch.nn.Conv2d):
            # print("This module is Conv")
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module=module, name='weight')
        else:
            # print("This module is something else...")
            pass

    return model



# @title FLOPS computation
# Code from https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/imagenet/l1-norm-pruning/compute_flops.py
import numpy as np
import os

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable


def print_model_param_nums(model=None):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))

def count_model_param_flops(model=None, input_res_width=480, input_res_height=320, multiply_adds=True):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.weight.data != 0).float().sum()
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(3,input_res_width,input_res_height).unsqueeze(0).to(model.parameters().__next__().device), requires_grad = True)
    out = model(input)


    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))

    print('Number of FLOPs: %.6f GFLOPs (%.2f MFLOPs)' % (total_flops / 1e9, total_flops / 1e6))

    return total_flops


def getfilesizeMB(path):
    size_of_file = os.path.getsize(path) / (1024 * 1024)  # Size in MB
    return size_of_file



#-------------------------- General PUBLIC VARIABLE --------------------------------------------

BBOX_COLOR = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]



#-------------------------- LOG RESULT VARIABLE -----------------------------------------------
time_inference_MPRnet = []
time_inference_YOLO = []



#---------------------- Check system -----------------------------------------
print(f"cuda available :: {torch.cuda.is_available()}")
os.makedirs(result_dir, exist_ok=True)
torch.manual_seed(42)







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

# print(summary(MPRNet_MODEL, input_size=(1, 3, 480, 320)))
count_model_param_flops(model=MPRNet_MODEL.eval(), input_res_width=480, input_res_height=320, multiply_adds=True)
print(f"model file size {getfilesizeMB(path=MPRNet_MODEL_WEIGHT)} MB.")


print("Pruning MPRNet Model ...")
mprnet_model_prepare_prune = MPRNet_MODEL
mprnet_model_after_prune = prune_model_MPRNet(model=mprnet_model_prepare_prune, amount=0.3)
print("MPRNet Model pruned")
MPRNet_MODEL = mprnet_model_after_prune
print("saving pruned MPRNet model ....")
torch.save(obj= MPRNet_MODEL.state_dict(), f='./MPRNet_trained_pruned.pth')
print("Pruned MPRNet Model saved.")

count_model_param_flops(model=MPRNet_MODEL.eval(), input_res_width=480, input_res_height=320, multiply_adds=True)
print(f"model file size {getfilesizeMB(path='./MPRNet_trained_pruned.pth')} MB.")


print("Quantizing MPRNet")
load_file = run_path(os.path.join("MPRNet.py"))
MPRNet_MODEL_PRUNED = load_file['MPRNet']()
MPRNet_MODEL_PRUNED.cuda()
MPRNet_MODEL_WEIGHT_PRUNED = './MPRNet_trained_pruned.pth'
MPRNet_MODEL_PRUNED.load_state_dict(torch.load(f=MPRNet_MODEL_WEIGHT_PRUNED, weights_only=True))


MPRNet_MODEL_QUANTIZED = quantize_dynamic(model=MPRNet_MODEL_PRUNED.cpu(),
                 qconfig_spec={nn.Linear, nn.Conv2d, nn.Sequential},
                 dtype=torch.qint8)
print("MPRNet Model quantized")

print("saving quantized MPRNet model ....")
torch.save(obj= MPRNet_MODEL_QUANTIZED.state_dict(), f='./MPRNet_trained_quantized.pth')
print("Quantized MPRNet Model saved.")




print("Initilize MPRNet ....")
load_file = run_path(os.path.join("MPRNet.py"))
MPRNet_MODEL_PRUNED_QUANTED = load_file['MPRNet']()
MPRNet_MODEL_PRUNED_QUANTED.cuda()


MPRNet_MODEL_WEIGHT_PRUNED_QUANTED = './MPRNet_trained_quantized.pth'

MPRNet_MODEL_PRUNED_QUANTED.load_state_dict(torch.load(f=MPRNet_MODEL_WEIGHT_PRUNED_QUANTED, weights_only=True))
# load_checkpoint(MPRNet_MODEL_PRUNED_QUANTED, MPRNet_MODEL_WEIGHT_PRUNED)

# print(summary(MPRNet_MODEL_PRUNED_QUANTED, input_size=(1, 3, 480, 320)))
count_model_param_flops(model=MPRNet_MODEL_PRUNED_QUANTED.eval(), input_res_width=480, input_res_height=320, multiply_adds=True)
print(f"model file size {getfilesizeMB(path=MPRNet_MODEL_WEIGHT_PRUNED_QUANTED)} MB.")

















#-----------------------   Initial USB Camera ------------------------------
# SOURCE_TYPE = 'usb'
# USB_IDX = 0
# RES_W, RES_H = 480, 320


# cap_arg = USB_IDX
# cap = cv2.VideoCapture(cap_arg)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
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
    # print(f"loop IDX :: {loop_idx}")
    im = cv2.imread(each_image, cv2.IMREAD_COLOR)
    if im.shape[:2] != (480, 320):
        # print("found image wrong shape ::", im.shape[:2])
        im = cv2.resize(src=im, dsize=(480, 320), dst=None, fx= None, interpolation=cv2.INTER_LINEAR)
    else:
        pass
        # print("image good shape", im.shape[:2])

    path_to_save_overwirte = os.path.join(INTERMEDITE_FOLDER, os.path.basename(each_image))
    # print(f"saving at {path_to_save_overwirte}")
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

        # frame = cv2.resize(frame, (RES_W, RES_H))
        
        # cv2.imshow(winname= 'USB Camera 00', mat=frame)
        # key = cv2.waitKey(5)

        # if key ==('q') or key == ord('Q'):
        #     break


#---------------------  Restoration -----------------------------


        time_start_retoration = time.perf_counter()     # start time stamp
        
        img = PIL.Image.open(each_image).convert('RGB')
        input_ = TF.to_tensor(img).unsqueeze(0).cuda()




        restored_image = MPRNet_MODEL_PRUNED_QUANTED(input_)       # inference




        #----------------   edit image ---------------------------------
        restored_image = restored_image[0]
        restored_image = torch.clamp(restored_image, 0, 1)

        restored_image = restored_image.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored_image = img_as_ubyte(restored_image[0])

        restored_image = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)



        time_finish_retoration = time.perf_counter()        # finish time stamp


        time_inference_elapse = time_finish_retoration - time_start_retoration
        print(f"Time Restoration :: {time_inference_elapse} seconds.")
        time_inference_MPRnet.append(time_inference_elapse)




#--------------------- YOLO Detect -----------------------------
        time_start_detect = time.perf_counter()        # start time stamp


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

        time_finish_detect = time.perf_counter()       # finish time stamp
        time_inference_elapse = time_finish_detect - time_start_detect
        print(f"Time Detection :: {time_inference_elapse} seconds.", end='\n\n')
        time_inference_YOLO.append(time_inference_elapse)


#-------------------- save image ----------------------
        path_to_save = os.path.join(result_dir, os.path.basename(each_image))
        cv2.imwrite(filename=path_to_save, img=restored_image)






#------------------------------  print log result   ---------------------------------

print("\n\nLog Result ...", end='\n\n')
time_inference_MPRNet_avg = sum(time_inference_MPRnet) / len(time_inference_MPRnet)
time_inference_YOLO_avg = sum(time_inference_YOLO) / len(time_inference_YOLO)

# print(summary(MPRNet_MODEL_PRUNED_QUANTED, input_size=(1, 3, 480, 320)))
# count_model_param_flops(model=MPRNet_MODEL_PRUNED_QUANTED.eval(), input_res_width=480, input_res_height=320, multiply_adds=True)
# YOLO_MODEL_PRUNED.info()

print(f"Time Inference MPRNet {time_inference_MPRnet}")
print(f"AVG :: {time_inference_MPRNet_avg} seconds.")
print("\n\n\n")
print(f"Time Inference YOLO {time_inference_YOLO}")
print(f"AVG :: {time_inference_YOLO_avg} seconds.")


