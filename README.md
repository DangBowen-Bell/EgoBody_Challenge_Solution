Code repository for our solution to the [EgoBody Challenge](https://codalab.lisn.upsaclay.fr/competitions/6351#learn_the_details).

In this repository, we explore some effective data augmentations to improve the model generalization of 3D human pose and shape estimation from a single egocentric image. 


## Install
Our code is mainly based on [SPIN](https://github.com/nkolot/SPIN), please refer to this repository for the installation of the environment.


## Data
You need to download the following data to start the experiment:

- [EgoBody](https://github.com/sanweiliti/EgoBody) dataset 
- Some essential data from [SPIN](https://github.com/nkolot/SPIN) (in the ```data``` directory)
- Pre-trained model from [EFT](https://github.com/facebookresearch/eft)

Then you need to specify their paths in ```config.py```.

You also need to generate the 2D keypoints for calculating 2D joint loss by running:

```
python keypoints.py
```

The generated 2D keypoints data will save as .npy file for easy loading.


## Train
You can train on the EgoBody dataset using pre-trained model by running:

```
python train.py --name exp_name --pretrained_checkpoint=/path/to/pre-tained/model.pt
```

The checkpoints and tensorboard files will be saved in the ```logs``` directory by default.

Please refer to the ```train_options.py``` for adding more data augmentations and setting other parameters.

You can download our best model [here](https://drive.google.com/file/d/1gBMyJkhMqlBGyUe7GA2NS6Jbz-XxYdTv/view?usp=sharing).

## Reference
The majority of this repository is borrowed from [SPIN](https://github.com/nkolot/SPIN). We also use some functions from [EFT](https://github.com/facebookresearch/eft) and [EgoBody](https://github.com/sanweiliti/EgoBody). Thank these authors for their great work.