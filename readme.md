# Combite MPRNet (derain) + YOLO11s
Not dont yet, can only Derain

still working on connect with USB camera

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

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


## reference
https://github.com/swz30/MPRNet

## note
Right now, cann't connect with usb camera yet. just scan image in folder.
MPRNet and YOLO not connect each other yet.