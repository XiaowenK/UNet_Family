# Introduction

This project will compared the existing UNet-like models with different open source dataset. My implementation is mainly based on Pytorch. This Respo will be a work in progress, and I will keep on updating it as many new models appear in the future.

Furthermore, for more details of UNet structure, please visit my [blog](https://xiaowenk.github.io/).

**WIP**





# Related Papers

* UNet : [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
* UNet++ : [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/pdf/1807.10165.pdf)
* Att_UNet : [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/pdf/1804.03999.pdf)
* ResUNet : [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
* RexUNet : [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
* Adversarial Learning : [Adversarial Learning for Semi-Supervised Semantic Segmentation](https://arxiv.org/pdf/1802.07934.pdf)





# Dataset

Datasets of this project:

- #### Retina Vessel : [Link](https://pan.baidu.com/s/1pOwiM4eXb-oDkbzSYx7rhg), keyword : jti3

This dataset consist 60 retina images from two subsets, [DRIVE](https://drive.grand-challenge.org/) and [STARE](https://cecas.clemson.edu/~ahoover/stare/), 40 and 20 images respectively. Please find more details about these two datasets in their official website, or you can directly download my preprocessed version with the link above.

During the data preprocess part, for DRIVE, we first center cropped the original size 565x584 to 528x576, then split a whole image into 132 small pathes (without overlaps) with size 48x48.

For STARE, the same pipeline were used, 700x605 to 672x576, split into 168 patches.

You can see the data preprocess pipeline step by step and how to split the training and testing set in the following scripts:

```
- utils/
    |--- RV_data_preprocess.py
    |--- gen_txt_RV.py
```



- #### **WIP**



If you are interested in the project, you can also compare the performance of different models by the following datasets:

- [Stanford Background Dataset](http://dags.stanford.edu/projects/scenedataset.html)
- [Sift Flow Dataset](http://people.csail.mit.edu/celiu/SIFTflow/)
- [Barcelona Dataset](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/)
- [Microsoft COCO dataset](http://mscoco.org/)
- [MSRC Dataset](http://research.microsoft.com/en-us/projects/objectclassrecognition/)
- [LITS Liver Tumor Segmentation Dataset](https://competitions.codalab.org/competitions/15595)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_road.php)
- [Pascal Context](http://www.cs.stanford.edu/~roozbeh/pascal-context/)
- [Data from Games dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)
- [Human parsing dataset](https://github.com/lemondan/HumanParsing-Dataset)
- [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)
- [Microsoft AirSim](https://github.com/Microsoft/AirSim)
- [MIT Scene Parsing Benchmark](http://sceneparsing.csail.mit.edu/)
- [COCO 2017 Stuff Segmentation Challenge](http://cocodataset.org/#stuff-challenge2017)
- [ADE20K Dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [INRIA Annotations for Graz-02](http://lear.inrialpes.fr/people/marszalek/data/ig02/)
- [Daimler dataset](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html)
- [ISBI Challenge: Segmentation of neuronal structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/)
- [INRIA Annotations for Graz-02 (IG02)](https://lear.inrialpes.fr/people/marszalek/data/ig02/)
- [Pratheepan Dataset](http://cs-chan.com/downloads_skin_dataset.html)
- [Clothing Co-Parsing (CCP) Dataset](https://github.com/bearpaw/clothing-co-parsing)
- [Inria Aerial Image](https://project.inria.fr/aerialimagelabeling/)
- [ApolloScape](http://apolloscape.auto/scene.html)
- [UrbanMapper3D](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17007&pm=14703)
- [RoadDetector](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17036&pm=14735)
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
- [Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/)





# Evaluation

In this project, we used k-fold CV to train and estimate the model performance.

```
# k-fold CV Algorithm

Step 1: split the whole dataset into equal K shares
Step 2: for i in K:
            take the #i share as test set
            for j in K:
                if j != i:
                    #j share as train set
            train, val, test
Step 3: Average K test results as the final score
```

And learning rate adjustment strategy is as shown belowed: 

If there is no decreasing in validation loss in 5 epoches, lr decrease by factor 2. This is implemented by **ReduceLROnPlateau** of Pytorch.

About the evaluation metric, we use **Dice Coefficient (DSC, %)** to evaluate the model performance. The defination of DSC is as belowed:

**WIP**

To be clearified, there is a slightly difference between DSC in testing part and Soft Dice Coefficient Loss in training part:



| RV(5-folds + lr_scheduler, bs=128) |   Retina Vessel   | Inria Aerial | Nodule Xray |
| ---------------------------------- | :---------------: | :----------: | :---------: |
| UNet (S)                           |   0.409 ± 0.033   |      -       |      -      |
| Att_UNet (S)                       |   0.402 ± 0.020   |      -       |      -      |
| UNet++ (S)                         |   0.378 ± 0.030   |      -       |      -      |
| ResUNet-50 (S)                     |   0.379 ± 0.021   |      -       |      -      |
| ResUNet-101 (S)                    |   0.379 ± 0.018   |      -       |      -      |
| ResUNet-101 (P)                    |   0.281 ± 0.012   |      -       |      -      |
| RexUNet-101 (P)                    | **0.280 ± 0.013** |      -       |      -      |
| Adv-RexUNet-101 (P)                |                   |      -       |      -      |

PS: **Att** for *Attention Gate*, **p** for *pretrained on ImageNet* and **Adv** for *adversarial learning*, **±** for standard deviation, **S** for *trained from scratch* and **P** for *pretrained on ImageNet*.

| RV(5-folds + lr_scheduler, bs=64) | Retina Vessel | Inria Aerial | Nodule Xray |
| --------------------------------- | :-----------: | :----------: | :---------: |
| UNet (S)                          | 0.358 ± 0.036 |      -       |      -      |
| Att_UNet (S)                      | 0.349 ± 0.020 |      -       |      -      |
| UNet++ (S)                        | 0.339 ± 0.029 |      -       |      -      |
| ResUNet-50 (S)                    | 0.275 ± 0.015 |      -       |      -      |
| ResUNet-101 (S)                   | 0.268 ± 0.015 |      -       |      -      |
| ResUNet-101 (P)                   | 0.264 ± 0.011 |      -       |      -      |
| RexUNet-101 (P)                   | 0.269 ± 0.014 |      -       |      -      |
| Adv-RexUNet-101 (P)               |               |      -       |      -      |





# How to use

### Dependencies

To install all needed dependencies, please run:

```
pip3 install -r requirements.txt
```

Please also install the [Nvidia apex module](https://github.com/NVIDIA/apex) to speeding up the training and saving GPU memory.

### Prepare the data

Please download the data from the link above and put them in the *database* folder to construct the following folder structure:

```
- database/
     |--- Retina_Vessel/
     |         |--- before_organized/
     |         |           |--- STARE/
     |         |           |      |--- stare-images.tar
     |         |           |      |--- labels-vk.tar
     |         |           |      |--- labels-ah.tar  
     |         |           |
     |         |           |--- DRIVE/ 
     |         |                  |--- datasets.zip 
     |         |
     |         |--- organized/
                        |--- 48x48/
                               |--- whole/
                               |      |--- raw/
                               |      |--- mask/
                               |  
                               |--- patch/
                                      |--- raw/
                                      |--- mask/ 
```

And Please download the pretrained model of ResUNext101's encoder from my [share](https://pan.baidu.com/s/1pFPsd-spgIbsCSsnnblbDQ), password: wp2n, then put it to the folder: <./models>.



### Training

- ###### Retina Vessel:

`python3 train_RV.py "UNet" False`

UNet, Att_UNet, UNet_PP, ResUNet, ResUNext

### Testing

###### Retina Vessel:

```
python3 test_RV.py "UNet" False
```

```
positional arguments:
   arch          model architecture: UNet | Att_UNet | UNet_PP | ResUNet50 | ResUNet101| 
                                     ResUNext101
   pretrained    if pretrained on ImageNet: True | False
```



# Reference

**WIP**



# License

The license is MIT.

