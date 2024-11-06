# VCoTTA: Variational Continual Test-Time Adaptation
Official code for Variational Continual Test-Time Adaptation.

We provide VCotta on the CIFAR10 task

## Prerequisite
Please create and activate the following conda envrionment. To reproduce our results, please kindly create and use this environment.

```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate vcotta 
```


## Classification Experiments
### CIFAR10-to-CIFAR10C-standard task
```bash
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/vcotta.yaml
```

