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



parser = argparse.ArgumentParser(description='Demo MPRNet')
parser.add_argument('--input_dir', default='./samples/input/', type=str, help='Input images')
parser.add_argument('--result_dir', default='./samples/output/', type=str, help='Directory for results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['Deblurring', 'Denoising', 'Deraining'])

args = parser.parse_args()

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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

task    = args.task
inp_dir = args.input_dir
out_dir = args.result_dir

print("Start Program")

os.makedirs(out_dir, exist_ok=True)

print(f"cuda available :: {torch.cuda.is_available()}")


print("scan file")

files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                + glob(os.path.join(inp_dir, '*.JPG'))
                + glob(os.path.join(inp_dir, '*.png'))
                + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")
else:
    print(f"found files  {len(files)}  files.")

print("Load Model MPRNet")
# Load corresponding model architecture and weights
load_file = run_path(os.path.join("MPRNet.py"))
model = load_file['MPRNet']()
model.cuda()

print("Load weight")
# weights = os.path.join(task, "pretrained_models", "model_"+task.lower()+".pth")
weights = './model_deraining.pth'

weights_size = os.path.getsize(weights)
print(f"weight size :: {weights_size / 1000} KiloBytes")
load_checkpoint(model, weights)


print("model start working")
model.eval()
with torch.no_grad():
    img_multiple_of = 8
    idx_loop = 0

    
    for file_ in files:
        print(f"{idx_loop} file name :: {file_}")

        time_start = time.perf_counter()

        print("load file")
        img = PIL.Image.open(file_).convert('RGB')
        input_ = TF.to_tensor(img).unsqueeze(0).cuda()


        print("edit h/w of file")
        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-h if h%img_multiple_of!=0 else 0
        padw = W-w if w%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')


        print("start restoration")
        
        restored = model(input_)
        restored = restored[0]
        restored = torch.clamp(restored, 0, 1)


        print("edit h/w image result")
        # Unpad the output
        restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        print("saving result image")
        f = os.path.splitext(os.path.split(file_)[-1])[0]
        save_img((os.path.join(out_dir, f+'.png')), restored)


        time_stop = time.perf_counter()

        print(f"inference Time = {time_stop - time_start}")


        idx_loop += 1

print(f"Files saved at {out_dir}")




model = YOLO('yolo11s.pt')
results = model.val(data="coco8.yaml")
print(f"mAP50-95 :: {results.box.map}")
