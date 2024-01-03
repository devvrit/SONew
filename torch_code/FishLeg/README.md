<h1>FishLeg</h1>

## Overview
This folder contains slight modifications to the official PyTorch implementation of the FishLeg optimizer as introduced in [Fisher-Legendre (FishLeg) optimization of deep neural networks](https://openreview.net/pdf?id=c9lAOPvQHS).<br />
We adapted the FishLeg Autoencodeer on MNIST official code. For more examples, refer to the [official code](https://github.com/mtkresearch/FishLeg) <br />
FishLeg is a learnt second-order optimization method that uses natural gradients and ideas from Legendre-Fenchel duality to learn a direct and efficiently evaluated model for the product of the inverse Fisher with any vector in an online manner. Thanks to its generality, we expect FishLeg to facilitate handling various neural network architectures. The library's primary goal is to provide researchers and developers with an easy-to-use implementation of the FishLeg optimizer and curvature estimator.
## Installation
FishLeg is written in Python, and only requires PyTorch > 1.8.<br />
To perform hyperparameter tuning, we used [nni](https://nni.readthedocs.io). <br />

## Usage
FishLeg requires minimal code modifications to introduce it in existing training scripts. 
```Python
from FishLeg import FishLeg, FISH_LIKELIHOODS

...
likelihood = FISH_LIKELIHOODS["FixedGaussian".lower()](sigma=1.0, device=device)

...

model = nn.Sequential(...).to(device)

optimizer = FishLeg(
        model,
        likelihood
        aux_loader,
        lr=eta_fl,
        eps=eps,
        beta=beta,
        weight_decay=1e-5,
        update_aux_every=10,
        aux_lr=aux_eta,
        aux_betas=(0.9, 0.999),
        aux_eps=1e-8,
        damping=damping,
        pre_aux_training=25,
        sgd_lr=eta_sgd,
        device=device,
    )

...
```

See [MNIST_fishleg.py](MNIST_fishleg) for an usage demostration. <br />
See the FishLeg [documentation](https://mtkresearch.github.io/FishLeg) for a detailed list of parameters.
 

## Citation
```
@article{garcia2022FishLeg,
  title={Fisher-Legendre (FishLeg) optimization of deep neural networks},
  author={Garcia, Jezabel R and Freddi, Federica and Fotiadis, Stathi1 and Li, Maolin and Vakili, Sattar, and Bernacchia, Alberto and Hennequin,Guillaume },
  journal={},
  year={2023}
}
```

## Using nni for hyperparameter tuning
We have included a config file with hyperparameter ranges to sweep from. There is a [seeded_tuner.py](seeded_tuner.py) file which maintains the algorithm that decides on the next set of hyperparameters to test, given the ones tested so far. In order to use nni, change the `is_nni` variable in [MNIST_fishleg.py](MNIST_fishleg) to True. <br />
nni can be run using the following command:

```
$ nnictl create --config fishleg_config.yml --port 8010
```

When running multiple runs, make sure to change the port.