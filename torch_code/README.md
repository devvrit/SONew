<h1>PyTorch Autoencoder on MNIST code</h1>

## Overview
We adapted the [Eva's](https://github.com/lzhangbv/eva) and [FishLeg](https://github.com/mtkresearch/FishLeg) official PyTorch Autoencoder on MNIST code to support their official implementation of Eva, KFAC, and FishLeg. Use [pytorch_autoencoder_nni.py](pytorch_autoencoder_nni.py) to run Eva, KFAC, and tridiag-SONew. For FishLeg, refer the FishLeg folder included. We support hyperparameter tuning using [nni](https://nni.readthedocs.io). <br />

## SONew
In order to run tridiag-SONew optimizer, use `python pytorch_autoencoder_nni.py --opt_name tds` and modify related hyperparameters.


## Using nni for hyperparameter tuning
We have included config files with sample hyperparameter ranges to sweep from. The [autoencoder_tds_config.yml](autoencoder_tds_config.yml) file can be used to tune tridiag-SONew, while [autoencoder_config_nni.yml](autoencoder_config_nni.yml) can be used to tune either of KFAC or EVA by making `kfac_name` as either `'kfac'` or `'eva'` respectively in the file. To run hyperparametr tuning on FishLeg, refer the [FishLeg](FishLeg) folder.<br />
There is a [seeded_tuner.py](seeded_tuner.py) file which maintains the algorithm that decides on the next set of hyperparameters to test, given the ones tested so far. In order to use nni, change the `is_nni` variable in [pytorch_autoencoder_nni.py](pytorch_autoencoder_nni) to True. <br />
nni can be run using the following command:

```
$ nnictl create --config autoencoder_tds_config.yml --port 8010
```

When running multiple nni hyperparameter optimization runs, make sure to change the port.

# Reproducing results from paper
The Autoencoder code used in [SONew](https://arxiv.org/abs/2311.10085) and [Eva](https://github.com/lzhangbv/eva)/[FishLeg](https://openreview.net/pdf?id=c9lAOPvQHS) differes slightly in the activation used. We (SONew) uses Tanh, while Eva uses ReLU. When we attempted to integrate Eva's code with our Autoencoder benchmark, the results were unsatisfactory; for instance, KFAC and FishLeg underperformed Adam, a benchmark that the authors themselves compared against. Given these results and to minimize modifications to the official code, we decided to test our optimizer directly on their provided autoencoder architecture. In order to reproduce results on the architecture used in SONew paper, just modify the activation and run hyperparameter tuning. <br />
Comparison between Eva, KFAC, FishLeg, and SONew is provided in the appendix of [SONew](https://arxiv.org/abs/2311.10085). This uses the Eva/FishLeg official implementation, and using tridiag-SONew optimizer in it.