# Dark-ISP: Low-Light RAW Image Processing for Object Detection

Official PyTorch implementation of the paper:

> Dark-ISP: Enhancing RAW Image Processing for Low-Light Object Detection

This repository provides:
- RAW image preprocessing pipeline
- End-to-end low-light ISP network
- MMDetection-based training framework
- LOD / NOD benchmark support
- Pretrained checkpoints and evaluation scripts

Built with:
- PyTorch
- MMCV
- MMDetection
- RAWPy
- LibRaw

## ⚙️ Dependencies and Installation:
Create the conda virtual environment with python version 3.8 and CUDA version 10.2
```bash
  conda create -n dark-isp python=3.8
  conda activate dark-isp
```
## 📦 Install the requirement
```bash
  cd mmdetection_github
  pip install -r requirements.txt 
```
### ⚠️ There are several libraries that require editable installation. 
All of them must be the same as the given in this repository. If some libraries update automatically during the installation process, they need to be uninstalled and reinstalled.
#### 1.mmcv
```bash
cd ../mmcv-2.1.0
pip install -v -e .
```
#### 2.mmengine
``` bash
cd ../mmengine-0.10.5
pip install -v -e .
```
#### 3.mmdet
```bash
cd ../mmdetection_github
pip install -v -e .
```
#### 4.LibRaw and rawpy
Unzip them:
```bash
cd ../downloads
unzip LibRaw-0.21.1.zip
unzip rawpy.zip
```
##### If you have `sudo` permission:
```bash
cd LibRaw-0.21.1
./configure
make
sudo make install
cd ../rawpy
RAWPY_USE_SYSTEM_LIBRAW=1 pip install -e .
```
##### Otherwise:
```
cd LibRaw-0.21.1
./configure --prefix=/path/to/your/directory 
make 
make install
```
Export environment variables:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/directory/lib/
export PKG_CONFIG_PATH=/path/to/your/directory/lib/pkgconfig:$PKG_CONFIG_PATH
```
Install rawpy:
```bash
cd ../rawpy
RAWPY_USE_SYSTEM_LIBRAW=1 pip install -e .
```
## 🗄 Data Peparation
The LOD dataset and orignal RAW images are uploaded [here](https://pan.baidu.com/s/1Ek-q_9DaSLceLD4dhOYQAA?pwd=2025) (Extraction code：2025).

To accelerate training, it is necessary to preprocess RAW format files into npz files. You need to replace the corresponding directory name in `LOD_RAW_preprocess.py`
```python
python mmdetection_github/LOD_RAW_preprocess.py
```
The processed npz file was also uploaded to the cloud disk.
## 🤖 Training and Evaluation
Modify the path information in the config files.
```python
python mmdetection_github/tools/train.py mmdetection_github/configs/LOD/VOCmetric/R_Net_denoise_50.py
```
## 📷 Test
Download the checkpoint in [Baidu Netdisk](https://pan.baidu.com/s/1i9xUtQFjoFxC5mIbxJcq9A?pwd=2025) (Extraction code：2025) or [Google Drive]https://drive.google.com/drive/folders/1ZTJEdGcbkFozS73L0BM8vudpCWR0iZ93?usp=sharing.
```python
python mmdetection_github/tools/test.py mmdetection_github/configs/LOD/VOCmetric/R_Net_denoise_50.py mmdetection_github/checkpoints/0.704_LOD.pth

python mmdetection_github/tools/test.py mmdetection_github/configs/NOD/NOD_Nikon_R_Net_denoise.py mmdetection_github/checkpoints/0.308_NOD_Nikon.pth

python mmdetection_github/tools/test.py mmdetection_github/configs/NOD/NOD_Sony_R_Net_denoise.py mmdetection_github/checkpoints/0.319_NOD_Sony.pth
```
