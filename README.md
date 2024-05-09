# Master project

## Setup

### Python

The requirements.txt file includes all the packages required for the project.

```sh
pip install -r requirements.txt
```

### Dataset

The KITTI-360 dataset can be accessed and downloaded from (http://www.cvlibs.net/datasets/kitti-360/).

## Training

To start training the conditional model, confirm the settings and parameters in training.py and run the following line:

```sh
accelerate launch train.py
```

- The training script is by default set to cache the dataset in the first run, therefore it will take longer the first time running the training script.


## Acknowledgements

- The Denoising diffusion probabilistic model and the Unet architecture is based on the implementation from [kazuto1011/r2dm](https://github.com/kazuto1011/r2dm).
