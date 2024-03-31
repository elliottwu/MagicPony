# Quick Start

### Extract DINO Features
```
cd extract_dino
python extract.py -c configs/config.yml
```

### Segmentation
Make sure you have installed SAM and GroundingDINO, and have downloaded the corresponding checkpoints (see L77-79 in `segmentation/segment.py`)
```
cd segmentation
python segment.py --source_dir /path/to/dataset/source/dir --target_dir /path/to/dataset/target/dir --category horse
```
