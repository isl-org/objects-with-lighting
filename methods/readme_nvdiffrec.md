# nvdiffrec Experiments

Please, find the modified code base at: https://github.com/sanskar107/nvdiffrec

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
conda create -n nvdiffrec python=3.9
conda activate nvdiffrec
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install ninja imageio PyOpenGL glfw xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
pip install open3d
imageio_download_bin freeimage
```

## Train, NVS, and Relighting

We provide example commands for running training, novel view synthesis (NVS)
and relighting.

```bash
scene="antman"

# train, relighting and novel views (pass mesh as bbox to calculate optimal mesh scale)
python train.py --nvs false --ref_mesh "data/llff_data/$scene/" --out_dir out/$scene --bbox data/dataset_raw/$scene/test/neus_mesh.ply --envmap_dir data/dataset_raw/$scene/test/
```

## Using NeuS mesh with nvdiffrec

```bash
# pass base_mesh to use NeuS mesh as fixed geometry
python train.py --nvs false --ref_mesh "data/llff_data/$scene/" --out_dir out/$scene --bbox data/dataset_raw/$scene/test/neus_mesh.ply --envmap_dir data/dataset_raw/$scene/test/ --base_mesh data/dataset_raw/$scene/test/neus_mesh.ply

```
