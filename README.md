# mydrq
Added a distracting background based on the drq source code. https://github.com/denisyarats/drq


## Requirements
The simplest way to install all required dependencies is to create an anaconda environment by running
```
conda env create -f conda_env.yml
```
Then install pytorch. see in https://pytorch.org/

After the instalation ends you can activate your environment with
```
conda activate drq
```

## Instructions
To train the DrQ agent on the `Cartpole Swingup` task run
```
python train.py env=cartpole_swingup
```
or modify config.yaml, then run
```
python train.py
```
