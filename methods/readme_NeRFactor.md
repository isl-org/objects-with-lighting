# NeRFactor Experiments

Please, find the modified code base at: https://github.com/sanskar107/nerfactor/tree/objrel

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
# convert to LLFF format
python scripts/create_nerf_data.py data/dataset_raw/$scene/test data/llff_data/$scene --overwrite
```

## Dependencies

Install the author's original dependencies.

```bash
conda env create -f environment.yml
conda activate nerfactor
```

## Train, NVS, and Relighting

We provide example commands for running training, novel view synthesis (NVS) and relighting. It consists of training brdf priors on merl dataset, training NeRF, extracting geometry buffers, shape pretraining, joint optimization, and relighting.

### Train BRDF prior

Train BRDF prior on merl dataset as described in the author's original readme.

```bash
gpus='0'
proj_root="<path/to/repo>"
repo_dir="<path/to/repo>"

data_root="$repo_dir/data/brdf_merl_npz/ims256_envmaph16_spp1"
outroot="$repo_dir/output/train/merl"
viewer_prefix=''
REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='brdf.ini' --config_override="data_root=$data_root,outroot=$outroot,viewer_prefix=$viewer_prefix"
```

### Convert LLFF data to NeRF format

```bash
scene_dir="data/llff_data/$scene"
h='512'
n_vali='9'

outroot="data/input/$scene"

REPO_DIR="$repo_dir" "$repo_dir/data_gen/nerf_real/make_dataset_run.sh" --scene_dir="$scene_dir" --h="$h" --n_vali="$n_vali" --outroot="$outroot"
```

### Train NeRF

```bash
near="0.03"
far="1.0"
data_root="data/input/$scene"
imh='512'

lr='5e-4'

outroot="data/output/train/${scene}_nerf"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='nerf.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,lr=$lr,outroot=$outroot,viewer_prefix=$viewer_prefix"
```

### Extract geometry buffers from trained NeRF

This is the slowest step as it extracts geometry buffers from all training, and validation poses. We spawn one job per camera view, and the index is indicated by `idx` variable. It is recommended to use array jobs to run this step, and replace `idx` with the array id.

```bash
data_root="data/input/$scene"
imh='512'
lr='5e-4'

trained_nerf="data/output/train/${scene}_nerf/lr$lr"
occu_thres='0.5'
scene_bbox=''

idx="<idx>"
echo "====== Index: $idx ======"

out_root="data/output/train/${scene}_geom"
mlp_chunk='475000' # bump this up until GPU gets OOM for faster computation

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/geometry_from_nerf_run.sh" "$gpus" --data_root="$data_root" --trained_nerf="$trained_nerf" --out_root="$out_root" --imh="$imh" --scene_bbox="$scene_bbox" --occu_thres="$occu_thres" --mlp_chunk="$mlp_chunk" --idx="$idx"
```

### Shape Pretraining and Joint Optimization

```bash
# Shape Pretraining
model="nerfactor"
overwrite='True'
gpus='0,1,2,3'
use_nerf_alpha='True'
surf_root="data/output/train/${scene}_geom"
shape_outdir="data/output/train/${scene}_shape"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='shape.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,outroot=$shape_outdir,viewer_prefix=$viewer_prefix,overwrite=$overwrite"

# Joint Optimization
shape_ckpt="$shape_outdir/lr1e-2/checkpoints/ckpt-2"
brdf_ckpt="$repo_dir/output/train/merl/lr1e-2/checkpoints/ckpt-50"

xyz_jitter_std=0.001

test_envmap_dir="data/llff_data/$scene/val_envmaps.npy"
shape_mode='finetune'
outroot="data/output/train/${scene}_${model}"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config="$model.ini" --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,shape_model_ckpt=$shape_ckpt,brdf_model_ckpt=$brdf_ckpt,xyz_jitter_std=$xyz_jitter_std,test_envmap_dir=$test_envmap_dir,shape_mode=$shape_mode,outroot=$outroot,viewer_prefix=$viewer_prefix,overwrite=$overwrite"
```

### Test Relighting and Novel Views

```bash
ckpt="$outroot/lr5e-3/checkpoints/ckpt-2"
color_correct_albedo='false'

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/test_run.sh" "$gpus" --ckpt="$ckpt" --color_correct_albedo="$color_correct_albedo"
```

## Using NeuS Mesh

We can extract geometry buffers from NeuS mesh and run shape training and joint optimization on the extracted buffers.

```bash
in_path="data/dataset_raw/$scene/test/"
outdir="data/output/train/${scene}_geom"

python geom_from_neus.py --input $in_path --out $outdir
```
