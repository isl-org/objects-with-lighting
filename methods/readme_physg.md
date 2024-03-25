# PhySG Experiments

Please, find the modified code base at: [https://github.com/Kai-46/PhySG-adapted](https://github.com/Kai-46/PhySG-adapted)

## Dataset

Download and extract our dataset like below:
```
dataset_all
├── antman
├── apple
├── chest
├── gamepad
├── ping_pong_racket
├── porcelain_mug
├── tpiece
└── wood_bowl
```

## Modify the data path in the following files
```
Line 8 of code/envmaps/fit_all_envmaps.py
Line 8 of code/fit_all.py
Line 8 of code/eval_all.py
```

## Set up environment and start training/evaluation
```
# set up conda env
. ./env.sh

# fit all envmaps with spherical gaussians
cd code/envmaps
python fit_all_envmaps.py

# train all scenes
cd code
python fit_all.py  # output is written to ../exps

# evaluate all scenes
cd code
python eval_all.py # output is gathered to ../evals/gather
```
