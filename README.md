---

<div align="center">    
 
# VAEs perform Independent Mechanism Analysis   

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/rpatrik96/ima-vae/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

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

# install pre-commit hooks
pre-commit install
 ```   
 Next, navigate to any file and run it.   
```bash
 python3 ima_vae/cli.py fit --help
 python3 ima_vae/cli.py fit --config configs/trainer.yaml --config configs/synth/moebius/moebius.yaml --config configs/synth/moebius/2d.yaml --model.prior=beta
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

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:



### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
