# InvRender Experiments
Please find the modified code base at: https://github.com/Tianhang-Cheng/InvRender_baseline

## Dataset
download our dataset and extract to 'dataset'
then transform our dataset to neus format
run script to put training data to 'dataset_neus'
put evaluation data to 'dateset_neus_test'
for synthetic4relight, bmvs and dtu dataset, the process is similar
```bash

cd datasets

python create_neus_data.py  path/to/antman/test  dataset_neus/antman
python create_neus_data.py  path/to/apple/test  dataset_neus/apple
python create_neus_data.py  path/to/chest/test  dataset_neus/chest
python create_neus_data.py  path/to/tpiece/test  dataset_neus/tpiece
python create_neus_data.py  path/to/gamepad/test  dataset_neus/gamepad
python create_neus_data.py  path/to/ping_pong_racket/test  dataset_neus/ping_pong_racket
python create_neus_data.py  path/to/porcelain_mug/test  dataset_neus/porcelain_mug
python create_neus_data.py  path/to/wood_bowl/test  dataset_neus/wood_bowl

python create_neus_data_test.py  path/to/antman/test  dataset_neus_test/antman
python create_neus_data_test.py  path/to/apple/test  dataset_neus_test/apple
python create_neus_data_test.py  path/to/chest/test  dataset_neus_test/chest
python create_neus_data_test.py  path/to/tpiece/test  dataset_neus_test/tpiece
python create_neus_data_test.py  path/to/gamepad/test  dataset_neus_test/gamepad
python create_neus_data_test.py  path/to/ping_pong_racket/test  dataset_neus_test/ping_pong_racket
python create_neus_data_test.py  path/to/porcelain_mug/test  dataset_neus_test/porcelain_mug
python create_neus_data_test.py  path/to/wood_bowl/test  dataset_neus_test/wood_bowl
```

## Training

you need to modify line 55 to line 86 of datasets/syn_dataset.py to assign the path of masks for different types of datasets

### geometry
```bash
python training/exp_runner.py \
    --conf confs_sg/default.conf \
    --data_split_dir /path/to/dataset_neus/antman \
    --expname antman \
    --trainstage IDR
```

### illumination
```bash
python training/exp_runner.py \
    --conf confs_sg/default.conf \
    --data_split_dir  /path/to/dataset_neus/antman \
    --expname antman \
    --trainstage Illum
```

### material
```bash
python training/exp_runner.py \
    --conf confs_sg/default.conf \
    --data_split_dir  /path/to/dataset_neus/antman \
    --expname antman \
    --trainstage Material
```

## Evaluation
### Novel views
Note: change the checkpoint path accordingly.

```bash
python training/exp_runner.py \
    --conf confs_sg/default.conf \
    --data_split_dir  /path/to/dataset_neus/antman \
    --expname antman \
    --trainstage Material \
    --is_continue \
    --eval 
```

### Relighting
Note: change the checkpoint path accordingly.
you may need to modify line 242 to line 273 of train_material.py for different types of datasets
for our dataset, you should run envmaps/fig_envmap_with_sg.py to create ground-truth envmap for relighting

```bash
python training/exp_runner.py \
    --conf confs_sg/default.conf \
    --data_split_dir  /path/to/dataset_neus/antman \
    --expname antman \
    --trainstage Material \
    --is_continue \
    --eval_relight
```
