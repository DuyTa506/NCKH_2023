# NCKH 2023 

Code for the paper NCKH 2023 


![framework](LPG-MI/imgsimgs/framework.jpg)

# Requirement

Install the environment as follows:

```bash
# create conda environment
conda create -n NCKH_2023  python=3.9
conda activate PLG_MI
# install pytorch 
conda install pytorch==1.10.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# install other dependencies
pip install -r requirements.txt
```

# Preparation

This code contains 2 scenarios :
 - Training model with Diffirential Privacy
 - Attack model by Model Inversion with cGAN method

## Dataset

- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  , [FFHQ](https://drive.google.com/open?id=1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv)
  and [FaceScrub](http://vintage.winklerbros.net/facescrub.html) are used for expriments (we
  use [this script](https://github.com/faceteam/facescrub) to download FaceScrub and some links are unavailable.)

- We follow the [KED-MI](https://github.com/SCccc21/Knowledge-Enriched-DMI/) to divide the CelebA into the private data
  and the public data. The private data of CelebA can be found
  at: https://drive.google.com/drive/folders/1uxSsbNwCKZcy3MQ4mA9rpwiJRhtpTas6?usp=sharing
- You should put them as follows:
    ```
    datasets
    ├── celeba
    │   └── img_align_celeba
    ├── facescrub
    │   └── faceScrub
    ├── ffhq
    │   └── thumbnails128x128
    └── celeba_private_domain
    ````

#Model Inversion :
### Models

- You can train target models following [KED-MI](https://github.com/SCccc21/Knowledge-Enriched-DMI/) or contact to duyfaker01@gmail.com for more details

- To calculate the KNN_dist, we get the features of private data on the evaluation model in advance. You can download
  at: https://drive.google.com/drive/folders/1Aj9glrxLoVlfrehCX2L9weFBx5PK6z-x?usp=sharing and put them in
  folder `./celeba_private_feats`.

### Top-n Selection Strategy

To get the pseudo-labeled public data using top-n selection strategy, pealse run the `top_n_selection.py` as follows:

```bash
python top_n_selection.py --model=VGG16 --data_name=ffhq --top_n=30 --save_root=reclassified_public_data
```

### Pseudo Label-Guided cGAN

To train the conditional GAN in stage-1, please run the `train_cgan.py` as follows:

```bash
python train_cgan.py \
--data_name=ffhq \
--target_model=VGG16 \
--calc_FID \
--inv_loss_type=margin \
--max_iteration=30000 \
--alpha=0.2 \
--private_data_root=./datasets/celeba_private_domain \
--data_root=./reclassified_public_data/ffhq/VGG16_top30 \
--results_root=PLG_MI_Results
```


### Image Reconstruction

To reconstruct the private images of specified class using the trained generator, pealse run the `reconstruct.py` as
follows:

```bash
python reconstruct.py \
--model=VGG16 \
--inv_loss_type=margin \
--lr=0.1 \
--iter_times=600 \
--path_G=./PLG_MI_Results/ffhq/VGG16/gen_latest.pth.tar \
--save_dir=PLG_MI_Inversion
```

# Examples of reconstructed face images

![examples](imgs/examples.jpg)

#Make private with diffirential privacy methods :
This Pytorch codebase implements efficient training of differentially private (DP) vision neural networks (CNN, including convolutional Vision Transformers), using [mixed ghost per-sample gradient clipping].

<p align="center">
  <img width="600" height="350" src="./assets/cifar10_memory_speed.png">
</p>

## ❓ What is this?
There are a few DP libraries that change the regular non-private training of neural networks to a privacy-preserving one. Examples include [Opacus](https://github.com/pytorch/opacus/blob/main/Migration_Guide.md#if-youre-using-virtual-steps), [FastGradClip](https://github.com/ppmlguy/fastgradclip), [private-transformers](https://github.com/lxuechen/private-transformers), and [tensorflow-privacy](https://github.com/tensorflow/privacy).

However, they are not suitable for DP training of large CNNs, because they are either not generalizable or computationally inefficient. E.g. causing >20 times memory burden or >5 times slowdown than the regular training.

<p align="center">
  <img width="750" height="250" src="./assets/cifar10_stress_tests.png">
</p>

This codebase implements a new technique --**the mixed ghost clipping**-- for the convolutional layers, that substantially reduces the space and time complexity of DP deep learning.

## 🔥 Highlights
* We implement a mixed ghost clipping technique for the Conv1d/Conv2d/Conv3d layers, that trains DP CNNs almost as light as (with 0.1%-10% memory overhead) the regular training. This allows us to train 18 times larger batch size on VGG19 and CIFAR10 than Opacus, as well as to train efficiently on ImageNet (224X224) or larger images, which easily cause out of memory error with private-transformers.
* Larger batch size can improve the throughput of mixed ghost clipping to be 3 times faster than existing DP training methods. On all models we tested, the slowdown is at most 2 times to the regular training.
* We support general optimizers and clipping functions. Loading vision models from codebases such as [timm](https://github.com/rwightman/pytorch-image-models) and [torchvision](https://pytorch.org/vision/stable/models.html), our method can privately train VGG, ResNet, Wide ResNet, ResNeXt, etc. with a few additional lines of code. 
* We demonstrate DP training of convolutional Vision Transformers (up to 300 million parameters, again 10% memory overhead and less than 200% slowdonw than non-private training). We improve from previous SOTA 67.4% accuracy to **83.0% accuracy at eps=1 on CIFAR100**, and to **96.7% accuracy at eps=1 on CIFAR10**.

<p align="center">
  <img width="750" height="300" src="./assets/cifar100_vit.png">
</p>

## :beers: Examples
To DP training models on CIFAR10 and CIFAR100, one can run
```bash
python -m cifar_DP --lr 0.001 --epochs 3 --model beit_large_patch16_224
```


Arguments:
- `--lr`: learning rate, default is 0.001
- `--epochs`: number of epochs, default is 1 
- `--model`: name of models in [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm/models), default is `resnet18`; see supported models below
- `--cifar_data`: dataset to train, `CIFAR10` (default) or `CIFAR100`
- `--eps`: privacy budget, default is 2
- `--grad_norm`: per-sample gradient clipping norm, default is 0.1 
- `--mode`: which DP clipping algorithm to use, one of `ghost_mixed`(default; the mixed ghost clipping), `ghost` (the ghost clipping), `non-ghost` (the Opacus approach), `non-private` (standard non-DP training)
- `--bs`: logical batch size that determines the convergence and accuracy, but not the memory nor speed; default is 1000 
- `--mini_bs`: virtual or physical batch size for the gradient accumulation, which determines the memory and speed of training; default is 50
- `--pretrained`: whether to use pretrained model from `timm`, default is True

As a consequence, we can privately train most of the models from `timm` (this list is non-exclusive):
```python
beit_base_patch16_224, beit_large_patch16_224, cait_s24_224, cait_xxs24_224, convit_base, convit_small, convit_tiny, convnext_base, convnext_large, crossvit_9_240, crossvit_15_240, crossvit_18_240, crossvit_base_240, crossvit_small_240, crossvit_tiny_240, deit3_base_patch16_224, deit_small_patch16_224, deit_tiny_patch16_224,

dla34, dla102, dla169, ecaresnet50d, ecaresnet269d, gluon_resnet18_v1b, gluon_resnet50_v1b, gluon_resnet152_v1b, gluon_resnet152_v1d, gluon_resnet152_v1s, hrnet_w18, hrnet_w48, ig_resnext101_32x8d, inception_v3, jx_nest_base, legacy_senet154, legacy_seresnet18, legacy_seresnet152, mixer_b16_224, mixer_l16_224, pit_b_224, pvt_v2_b1,

resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, res2net50_14w_8s, res2next50, resnest50d, seresnet50, seresnext50_32x4d, ssl_resnet50, ssl_resnext50_32x4d, swsl_resnet50, swsl_resnext50_32x4d, tv_resnet152, tv_resnext50_32x4d, twins_pcpvt_base, twins_pcpvt_large, twins_svt_base, twins_svt_large   

vgg11, vgg11_bn, vgg13, vgg16, vgg19, visformer_small, vit_base_patch16_224, vit_base_patch32_224, vit_large_patch16_224, vit_small_patch16_224, vit_tiny_patch16_224, volo_d1_224, wide_resnet50_2, wide_resnet101_2, xception, xcit_large_24_p16_224, xcit_medium_24_p16_224, xcit_small_24_p16_224, xcit_tiny_24_p16_224
```
We also support models in `torchvision` and other vision libraries, e.g. `densenet121, densnet161, densenet201`.
Or you can build your own Deep Learning model .

<!--
# :warning: Caution
* **Batch normalization does not satisfy DP.** This is because the mean and variance of batch normalization is computed from data without privatization. To train DP networks, replace batch normalization with group/instance/layer normalization. [Opacus (>v1.0)](https://github.com/pytorch/opacus/blob/main/tutorials/guide_to_module_validator.ipynb) provides an easy fixer for this replacement via `opacus.validators.ModuleValidator.fix`, but you can also change the normalization layer manually. 
* **Extra care needed for sampling.** Taking virtual step with fixed virtual batch size is not compatible with Poisson sampling. [Opacus] provides `BatchMemoryManager` to feature this [sampling issue](https://github.com/pytorch/opacus/blob/main/Migration_Guide.md#if-youre-using-virtual-steps) and our mixed ghost clipping can be merged

## Acknowledgement
This code is largely based on https://github.com/lxuechen/private-transformers (v0.1.0) and https://github.com/pytorch/opacus (v0.15).
