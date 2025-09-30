# Combite MPRNet (derain) + YOLO11s
Not dont yet, can only Derain

still working on connect with USB camera

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependecies.

```bash
conda create --name venv_mprnet python=3.8
conda activate venv_mprnet
pip install numpy ultralytics
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

## Example to test code library
```bash
python demo_yolo.py --task Deraining --input_dir ./Images/Raw_images/ --result_dir ./Images/Result_images/
```

## Example code for run MPRNet_Yolo_ver1.py
For now, can ONLY Derain, read image from folder, YOLO can drawing the rectangle prediction at result images, Pruning YOLO Model.

```bash
python MPRNet_Yolo_ver1.py --input_dir ./Images/Raw_images --result_dir ./Images/Result_images/ --task Deraining
```
Pruning MPRNet, connecting with USB Camera is underdevelopment . . .

## Example code for run MPRNet_Yolo_Camera.py
**Pipeline : USB Camera => MPRNet(pruned+quantized) + YOLO(pruned) => show detect image via opencv (not save to folder)**

can choose a task 'deblurring', 'denoising', 'deraining'

can choose USB Camera Index e.g. 0, 1, 2

can choose image resolution e.g. 720x480, 480x320
```bash
python MPRNet_Yolo_Camera.py --task deblurring --usb_cam_idx 2 --img_res 720x480
```



## reference
https://github.com/swz30/MPRNet

## note
Right now, cann't connect with usb camera yet. just scan image in folder.
MPRNet and YOLO not connect each other yet.