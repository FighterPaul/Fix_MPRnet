# Combite MPRNet (derain) + YOLO11s
Not dont yet

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
conda create --name venv_mprnet python=3.8
conda activate venv_mprnet
pip install numpy ultralytics
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

## simple test code
```bash
python demo_yolo.py --task Deraining --input_dir ./Images/Raw_images/ --result_dir ./Images/Result_images/
```


## reference
https://github.com/swz30/MPRNet

## note
Right now, cann't connect with usb camera yet. just scan image in folder.
MPRNet and YOLO not connect each other yet.