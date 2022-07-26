# MUTR3D: A Multi-camera Tracking Framework via 3D-to-2D Queries.  [Paper](https://arxiv.org/abs/2205.00613) - [Project Page](https://tsinghua-mars-lab.github.io/mutr3d/)


This repo implements the paper MUTR3D: A Multi-camera Tracking Framework via 3D-to-2D Queries. We built our implementation upon MMdetection3D.

The major part of the code is in the directory `plugin/track`. To use this code with MMDetection3D, we need older versions of MMDetection3D families(see Environment section), and you need to replace `mmdet3d/api` with the `mmdet3d/api` provided here. 


## How to run



## Environment
```
Python 3.8
PyTorch: 1.10.2+cu113
TorchVision: 0.11.3+cu113
pip uninstall torch
pip uninstall torchversion
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu113
OpenCV: 4.5.5
MMCV: 1.3.16
MMCV Compiler: GCC 9.4
MMCV CUDA Compiler: 11.3
MMDetection: 2.12.0
MMDetection3D: 0.13.0+b255d16

使用pip uninstall torch,torchvision
然后再安装这两个，不然后面会报错，也是直接用pip install 安装就好
opencv也直接卸载，然后等安装了mmdet3d之后，他会自己安装一个最新版本，你再下载相关的contribute的模块包
```



First, install: 
1. mmcv==1.3.14   #这里要用mmcv-full==1.3.14 docker中使用1.3.16也是匹配的，需要装1.3.16最好
这个需要使用官方的语句进行安装，要和相关的cuda和pytorch 对应起来，不然后面也会报错，类似这种，原先的语句我忘记了，自己进行测试看看
```
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

2. mmdetection==2.12.0    # mmdet==2.12.0
```
pip install mmdet==2.12.0
```
3. [nuscenses-devkit](https://github.com/nutonomy/nuscenes-devkit)    # nuscenes-devkit
```
pip install nuscenes-devkit
```
4. Note: for tracking we need to install:
`motmetrics==1.1.3`, not newer version, like `motmetrics==1.2.0`!!   # motmetrics==1.1.3
```
pip install motmetrics==1.1.3
```


opencv的问题，docker请使用opencv-python-headless


Second, clone mmdetection3d==0.13.0, but replace its `mmdet3d/api/` from mmdetection3d by `mmdet3d/api/` in this repo.

e.g. 
```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.13.0
# cp -r ../mmdet3d/api mmdet3d/
# cp ../mmdet3d/models/builder.py mmdet3d/models/
# cp ../mmdet3d/models/detectors/mvx_two_stage.py mmdet3d/models/detectors/mvx_two_stage.py

# replace the mmdetection3d/mmdet3d with the mmdet3d_full
rm -r mmdet3d/
cp -r ../mmdet3d_full ./mmdet3d

cp -r ../plugin ./ 
cp -r ../tools ./ 
# then install mmdetection3d following its instruction. 
# and mmdetection3d becomes your new working directories. 

修改setup.py
添加
make_cuda_ext(
                name='iou_loss_ext',
                module='mmdet3d.ops.iou_loss',
                sources=['src/sort_vert.cpp'],
                sources_cuda=['src/sort_vert_kernel.cu'])



pip install -v -e .
```

```
opencv的问题，docker请使用opencv-python-headless
pip install opencv-python-headless==4.5.5.62
pip install opencv-contrib-python==4.5.5.62
```


### Dataset preprocessing
After preparing the nuScenes Dataset following mmdetection3d,  you need to generate a meta file or say `.pkl` file. 

```
python3 tools/data_converter/nusc_track.py
```


### Run training

I provide a template config file in `plugin/track/configs/resnet101_fpn_3frame.py`. You can directly use this config or read this file, especially its comments, and modify whatever you want. I recommend using DETR3D pre-trained models or other nuScenes 3D Detection pre-trained models. 

basic training scripts on a machine with 8 GPUS: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train_tracker.sh plugin/track/configs/resnet101_fpn_3frame.py 8 --work-dir=work_dirs/experiment_name
```

basic test scripts
```
# You can perform inferece, then save the result file
python3 tools/test.py plugin/track/configs/resnet101_fpn_3frame.py <model-path> --format-only --eval-options jsonfile_prefix=<dir-name-for-saving-json-results>

# or you can perform inference and directly perform the evaluation
python3 tools/test.py plugin/track/configs/resnet101_fpn_3frame.py <model-path>  --eval --bbox
```

### Visualization
For visualization, I suggest user to generate the results json file first. I provide some sample code at `tools/nusc_visualizer.py` for visualizing the predictions, see function `_test_pred()` in `tools/nusc_visualize.py` for examples. 

## Results

| Backbones  | AMOTA-val | AMOTP-val | IDS-val | Download |   
|---|---|---| --- | --- |
| ResNet-101 w/ FPN  | 29.5  | 1.498 | 4388 | [model](https://drive.google.com/file/d/1MXbHWalo-zyt9TU31x-re4wOuX5G4wOH/view?usp=sharing) \| [val results](https://drive.google.com/file/d/1qf8D3cTDCdlspOEJpgRXnSGW9AotaRxP/view?usp=sharing)  |
| ResNet-50 w/ FPN  |  25.2 |  1.573| 3899 | [model](https://drive.google.com/file/d/1_BPDvDPKN7j476w2g5IMAagCW5szfF2y/view?usp=sharing) \| [val results](https://drive.google.com/file/d/1bIgsRgBwTjcGzlNaHAoXMCLWsqEr3cnH/view?usp=sharing)  |



## Acknowledgment

For the implementation, we rely heavily on [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d),[MOTR](https://github.com/megvii-model/MOTR), and [DETR3D](https://github.com/WangYueFt/detr3d)



## Relevant projects 
1. [DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](https://tsinghua-mars-lab.github.io/detr3d/)
2. [FUTR3D: A Unified Sensor Fusion Framework for 3D Detection](https://tsinghua-mars-lab.github.io/futr3d/)
3. For more projects on Autonomous Driving, check out our camera-centered autonomous driving projects page [webpage](https://tsinghua-mars-lab.github.io/vcad/) 


## Reference


```
@article{zhang2022mutr3d,
  title={MUTR3D: A Multi-camera Tracking Framework via 3D-to-2D Queries},
  author={Zhang, Tianyuan and Chen, Xuanyao and Wang, Yue and Wang, Yilun and Zhao, Hang},
  journal={arXiv preprint arXiv:2205.00613},
  year={2022}
}
```

Contact: [Tianyuan Zhang](http://tianyuanzhang.com/) at: `tianyuaz@andrew.cmu.edu` or `tianyuanzhang1998@gmail.com`
