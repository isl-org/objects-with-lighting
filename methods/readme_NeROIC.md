# NeROIC Experiments

Please, find the modified code base at: https://github.com/sanskar107/NeROIC

## Dataset

Download and extract the dataset as follows:

```bash
# Folder structure
# data/dataset_raw
# ├── bmvs
# │   ├── bear
# │   ├── clock
# │   ├── dog
# │   ├── durian
# │   ├── jade
# │   ├── man
# │   ├── sculpture
# │   └── stone
# ├── dtu
# │   ├── scan37
# │   ├── scan40
# │   ├── scan55
# │   ├── scan63
# │   ├── scan65
# │   ├── scan69
# │   ├── scan83
# │   └── scan97
# ├── objrel
# │   ├── antman
# │   ├── apple
# │   ├── chest
# │   ├── gamepad
# │   ├── ping_pong_racket
# │   ├── porcelain_mug
# │   ├── tpiece
# │   └── wood_bowl
# └── synth4relight_subsampled
#     ├── air_baloons
#     ├── chair
#     ├── hotdog
#     └── jugs
```

## Data Conversion

Convert the raw data to LLFF format using `scripts/create_nerf_data.py` from `object-relighting-dataset` repository.

```bash
python scripts/create_nerf_data.py data/dataset_raw/$scene/test data/llff_data/$scene --overwrite
```

## Dependencies

Install the author's original dependencies.

```bash
conda env create -f environment.yml
conda activate neuralpil
```

## Train, NVS, and Relighting

We provide example commands for running training, novel view synthesis (NVS)
and relighting.

```bash
scene="antman"
rwfactor='2'

datadir="data/llff_data/$scene"
envdir="data/llff_data/$scene/val_envmaps.npy"
expname=$scene

# train geometry
python train.py --config configs/${scene}_geometry.yaml --datadir data/llff_data/$scene

# extract normals
python generate_normal.py --config configs/${scene}_geometry.yaml --ft_path out_geometry/${scene}_geometry/epoch=29.ckpt --datadir data/llff_data/$scene

# train material
python train.py --config configs/${scene}_rendering.yaml --ft_path out_geometry/${scene}_geometry/epoch=29.ckpt --datadir data/llff_data/$scene


# test relighting
python test_relighting.py --config configs/${scene}_rendering.yaml --ft_path out_rendering/${scene}_rendering/epoch=9.ckpt --datadir data/llff_data/$scene --test_env_filename data/llff_data/$scene/val_envmaps.npy


# test novel views
python test_nvs.py --config configs/${scene}_rendering.yaml --ft_path out_rendering/${scene}_rendering/epoch=9.ckpt --datadir data/llff_data/$scene
```
