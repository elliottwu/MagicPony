# 🎠 MagicPony: Learning Articulated 3D Animals in the Wild (CVPR 2023)
#### [Project Page](https://3dmagicpony.github.io/) | [Video](https://youtu.be/KoLzpESstLk) | [Paper](https://arxiv.org/abs/2211.12497)


[Shangzhe Wu](https://elliottwu.com/)\*, [Ruining Li](https://ruiningli.com/)\*, [Tomas Jakab](https://www.robots.ox.ac.uk/~tomj)\*, [Christian Rupprecht](https://chrirupp.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi) (*equal contribution)

Visual Geometry Group, University of Oxford

## Setup (with [conda](https://docs.conda.io/en/latest/))

### 1. Install dependencies
```
conda env create -f environment.yml
```
or manually:
```
conda install -c conda-forge setuptools=59.5.0 numpy=1.23.1 matplotlib=3.5.3 opencv=4.6.0 pyyaml=6.0 tensorboard=2.10.0 trimesh=3.9.35 configargparse=1.5.3 einops=0.4.1 moviepy=1.0.1 ninja=1.10.2 imageio=2.21.1 pyopengl=3.1.6 gdown=4.5.1
pip install glfw xatlas
```

### 2. Install [PyTorch](https://pytorch.org/)
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
*Note*: The code is tested with PyTorch 1.10.0 and CUDA 11.3.

### 3. Install [NVDiffRec](https://github.com/NVlabs/nvdiffrec) dependencies
```
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn@v1.6#subdirectory=bindings/torch
imageio_download_bin freeimage
```
*Note*: The code is tested with tinycudann=1.6 and it requires GCC/G++ > 7.5 (conda's gxx also works: `conda install -c conda-forge gxx_linux-64=9.4.0`).

## Data
The preprocessed datasets can be downloaded using the scripts in `data/`, including birds, horses, giraffes, zebras and cows:
```
cd data
sh download_horse_combined.sh
sh download_horse_videos.sh
sh download_bird_videos.sh
sh download_giraffe_coco.sh
sh download_zebra_coco.sh
sh download_cow_coco.sh
```
*Note*: `horse_combined` consists of `horse_videos` from [DOVE](https://dove3d.github.io/) and additional images from [Weizmann Horse Database](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database), [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/), and [Horse-10](http://www.mackenziemathislab.org/horse10).

## Pretrained Models
The pretrained quadruped models can be downloaded using the scripts in `results/`:
```
cd results/quadrupeds
sh download_pretrained_horse.sh
sh download_pretrained_giraffe.sh
sh download_pretrained_zebra.sh
sh download_pretrained_cow.sh
```
as well as the pretrained bird model:
```
cd results/birds
sh download_pretrained_bird.sh
```

## Run
### Train and Testing
Check the configuration files in `config/` and run, eg:
```
python run.py --config config/quadrupeds/train_horse.yml --gpu 0 --num_workers 4
python run.py --config config/quadrupeds/test_horse.yml --gpu 0 --num_workers 4
```

## TODO
- [ ] Visualization scripts
- [ ] Evaluation scripts
- [ ] Data preprocessing scripts

## Citation
```
@InProceedings{wu2023magicpony,
  author    = {Shangzhe Wu and Ruining Li and Tomas Jakab and Christian Rupprecht and Andrea Vedaldi},
  title     = {{MagicPony}: Learning Articulated 3D Animals in the Wild},
  booktitle = {CVPR},
  year      = {2023}
}
```