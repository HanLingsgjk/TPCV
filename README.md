# TPCV
code for CVPR23 Paper Learning optical expansion from scale matching
## Requirements
The code has been tested with PyTorch 1.11.0 and Cuda 11.3.
```Shell
conda create -n tpcv python=3.9
conda activate tpcv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Compile CUDA code for inverse optical flow
```Shell
cd inverse_cuda
python setup.py install
```

Also need to install via pip
```Shell
pip install matplotlib
pip install opencv-python
...
```
## Dataset Configuration
To evaluate/train TPCV, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

We recommend manually specifying the path in `dataset_exp_orin.py`, like `def  __init__(self, aug_params=None, split='kitti_test', root='/new_data/datasets/KITTI/training',get_depth=0):` , because the automatic one often makes mistakes

You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder
```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```
## Pretrained weights on KITTI
https://drive.google.com/drive/folders/1Ddh1HYKVo5CVITmLfV9_8uJKMI3HdePd?usp=share_link

## Train on KITTI

```Shell
python train.py --name raft-kitti_3D --stage kitti --validation kitti --restore_ckpt ../TPCV/checkpotins/kitti_3D_flow.pth --gpus 0 --num_steps 60000 --batch_size 2 --lr 0.000125 --image_size 320 960 --wdecay 0.0001 --gamma=0.85
```

## Test on KITTI

```Shell
python dc_flow_eval.py --model=../TPCV/checkpotins/kitti_3D_flow.pth --mixed_precision --start=0
```
