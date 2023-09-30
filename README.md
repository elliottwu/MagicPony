# 🎠 MagicPony: Learning Articulated 3D Animals in the Wild (CVPR 2023)
#### [Project Page](https://3dmagicpony.github.io/) | [Video](https://youtu.be/KoLzpESstLk) | [Paper](https://arxiv.org/abs/2211.12497)


[Shangzhe Wu](https://elliottwu.com/)\*, [Ruining Li](https://ruiningli.com/)\*, [Tomas Jakab](https://www.robots.ox.ac.uk/~tomj)\*, [Christian Rupprecht](https://chrirupp.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi) (*equal contribution)

Visual Geometry Group, University of Oxford


https://github.com/elliottwu/MagicPony/assets/15921740/2e2f94a4-b16e-458e-b5ac-b69f6249c50d


This method learns single-image 3D reconstruction models of articulated animal categories, from just online photo collections, without any 3D geometric supervision or template shapes. 

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

### 4. Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) (for visulaization)
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

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

### Tetrahedral Grids
Download the tetrahedral grids:
```
cd data/tets
sh download_tets.sh
```

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
The test script saves some visualizations, including reconstruction images from input view, viewpoint and the resulting mesh in `.obj` format. For more visualizations, use the `scripts/visualize_results.py` as explained below.

## Visualize Results
Check `scripts/visualize_results.py` and run, eg:
```
python scripts/visualize_results.py \
--input_image_dir path/to/folder/that/contains/input/images \
--config path/to/the/test/config.yml \
--checkpoint_path path/to/the/pretrained/checkpoint.pth \
--output_dir folder/to/save/the/renderings \
--render_modes input_view other_views
```
`--input_image_dir` should contain test images named as `*_rgb.*`.

Supported `--render_modes` include:
- `input_view`: image rendered from the input viewpoint of the reconstructed textured mesh, shading map, gray shape visualization
- `other_views`: image rendered from 12 viewpoints rotating around the vertical axis of the reconstructed textured mesh, gray shape visualization
- `rotation`: video rendered from continuously rotating viewpoints around the vertical axis of the reconstructed textured mesh, gray shape visualization 
- `animation` (only suported for horses): two videos rendered from both a side viewpoint and continously rotating viewpoints of the reconstructed textured mesh animated using a sequence of pre-configured articulation parameters. `--arti_param_dir` is required which should contain a sequence of keyframe articulation parameters. By default, it points to `./scripts/animation_params` which contains a set of pre-computed parameters.
- `canonicalization` (only suported for horses): video of the reconstructed textured mesh morphing from the input pose to a pre-configured canonical pose


### Test-time Texture Finetuning
To enable test time texture finetuning, use the flag `--finetune_texture`, and (optionally) adjust the number of finetune iterations `--finetune_iters` and learning rate `--finetune_lr`.

For more precise texture optimization, provide instance masks in the same folder as `*_mask.png`. Otherwise, the background pixels might be pasted onto the object if shape predictions are not perfect aligned.


## Evaluation
We evaluate our method quatitatively with the keypoint transfer metric on the PASCAL dataset, following [A-CSM](https://arxiv.org/abs/2004.00614). To compute the metric, we first extract the mesh vertices with `scripts/visualize_results.py`, with a slight abuse of the script:
```
python scripts/visualize_results.py \
--input_image_dir path/to/preprocessed/pascal/images \
--config path/to/the/test/config.yml \
--checkpoint_path path/to/the/pretrained/checkpoint.pth \
--output_dir folder/to/save/the/output \
--evaluate_keypoint
```
Note we specified the `--evaluate_keypoint` flag.

Then, use the following to compute the keypoint transfer metric:
```
python evaluation/evaluate.py \
--acsm_annotations_root path/to/acsm/annotation/root \
--pascal_category horse \
--data_dir_test path/to/preprocessed/pascal/images \
--predictions_test_dir the/output/dir/from/the/previous/command \
--box_pad_frac 0.05
```
The code should automatically download the A-CSM annotations to the specified path, providing the user has write access to the path. If the script fails to download 
the annotations, they can be downloaded manually via [Dropbox](https://www.dropbox.com/s/3tj037gnk4gz11t/cachedir.tar?dl=1). The `--acsm_annotations_root` should be the path to the
folder that contains the extracted `cachedir` folder from the tar file.

As A-CSM forgot to correct for their box padding, we add the 0.05 here for consistency.

## TODO
- [x] Test time texture finetuning
- [x] Novel view visualization script
- [x] Animation visualization script
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
