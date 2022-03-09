# Generative Planning Method (ICLR 2022 Spotlight)


## Introduction
This is the PyTorch implementation of the Generative Planning Method (GPM) proposed in ["Generative Planning for Temporally Coordinated Exploration in Reinforcement Learning"](https://openreview.net/forum?id=YZHES8wIdE).
More information can be found on the project page: https://sites.google.com/site/hczhang1/projects/generative-planning.




## Usage
### Installation
The training environment (PyTorch and dependencies) can be installed as follows:
```
git clone --recursive https://github.com/Haichao-Zhang/generative-planning.git
cd generative-planning

python3 -m venv .venv
source .venv/bin/activate

pip install -e . -e alf

```



### Training
```
cd gpm/examples/
CUDA_VISIBLE_DEVICE=0 python3 -m alf.bin.train --gin_file=gpm_pendulum.gin --root_dir=EXP_PATH
```



## Cite

Please cite our work if you find it useful:

```
@inproceedings{generative_planning,
    author = {Haichao Zhang and Wei Xu and Haonan Yu},
    title  = {Generative Planning for Temporally Coordinated Exploration in Reinforcement Learning},
    booktitle = {International Conference on Learning Representations},
    year = {2022}
}
```


## Contact

For questions related to generative planning, please send me an email: ```hczhang1@gmail.com```