---

<div align="center">    
 
# Embrace the Gap: VAEs perform Independent Mechanism Analysis   

[//]: # ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.2206.02416-B31B1B.svg&#41;]&#40;https://arxiv.org/abs/2206.02416&#41;)

[![Conference](http://img.shields.io/badge/NeurIPS-2022-4b44ce.svg)](https://openreview.net/forum?id=G4GpqX4bKAH)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/ICLR-2019-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;  )

[![Paper](http://img.shields.io/badge/arxiv-stat.ML:2206.02416-B31B1B.svg)](https://arxiv.org/abs/2206.02416)

![CI testing](https://github.com/rpatrik96/ima-vae/workflows/CI%20testing/badge.svg?branch=master&event=push)
[![DOI](https://zenodo.org/badge/431811003.svg)](https://zenodo.org/badge/latestdoi/431811003)

<!--  
Conference   
-->   
</div>
 
## Description   
This is the code for the paper _Embrace the Gap: VAEs perform Independent Mechanism Analysis_, showing that optimizing the ELBO is equivalent to optimizing the IMA-regularized log-likelihood under certain assumptions (e.g., small decoder variance).

## How to run   
First, install dependencies   
```bash
# clone ima_vae   
git clone --recurse-submodules https://github.com/rpatrik96/ima-vae

# if forgot to pull submodules, run
git submodule update --init

# install ima_vae   
cd ima-vae
pip install -e .   
pip install -r requirements.txt

# install spriteworld
pip install -e ./spriteworld

# install submodule requirements
pip install --requirement ima/requirements.txt --quiet
pip install --requirement tests/requirements.txt --quiet
pip install --requirement spriteworld/requirements.txt --quiet

# install pre-commit hooks (only necessary for development)
pre-commit install
 ```   
 Next, navigate to the `ima-vae` directory and run `ima_vae/cli.py.   
```bash
 python3 ima_vae/cli.py fit --help
 python3 ima_vae/cli.py fit --config configs/trainer.yaml --config configs/synth/moebius.yaml --model.prior=beta
```

### Hyperparameter optimization

First, you need to log into `wandb`
```bash
wandb login #you will find your API key at https://wandb.ai/authorize
```

Then you can create and run the sweep
```bash
wandb sweep sweeps/synth/mlp/finding_optimal_gamma_uniform.yaml  # returns sweep ID
wandb agent <ID-comes-here> --count=<number of runs> # when used on a cluster, set it to one and start multiple processes
```



## Citation   

```
@inproceedings{
 reizinger_embrace_2022,
 title={Embrace the Gap: {VAE}s Perform Independent Mechanism Analysis},
 author={Patrik Reizinger and Luigi Gresele and Jack Brady and Julius Von K{\"u}gelgen and Dominik Zietlow and Bernhard Sch{\"o}lkopf and Georg Martius and Wieland Brendel and Michel Besserve},
 booktitle={Advances in Neural Information Processing Systems},
 editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
 year={2022},
 url={https://openreview.net/forum?id=G4GpqX4bKAH}
}

```   
