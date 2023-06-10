# Image Reconstruction from Deep Diffractive Neural Network 
## Abstract
In this project we will present research in the field of deep learning and image reconstruction.
The goal of the project was to understand whether it is possible to recover an image from a noise image passed through a deep optical network.
First we reproduced an Optical Network based on the article [To image, or not to image: Class-specific diffractive cameras with all-optical erasure of undesired objects](https://arxiv.org/abs/2205.13122) that demonstrates the use of a deep optical network as a classifying camera which transfers the image of the target class in high quality and for the other classes transfers a noise image.
In the second step, we recovered the original images from the noise images.
At the stage of reproducing the article, we built a model from free space propagation layers and phase mask layers, from which the model is composed in the article.
Next, we trained the network on the "MNIST" dataset and got similar results to the results of the article in the simulation of the optical network.
In the reconstruction phase, we wrote an algorithm which classifies the noise images into the original classes and we reached an average accuracy of 89.2% on 900 samples.
These results were a proof of feasibility that there is indeed information in the noise images. Last, we used a DNN with U-Net architecture.
This architecture consists of an encoder-decoder structure with skip connections.
We trained the network for image-to-image translation task.
The goal was to reconstruct the original images from the noise images, and we got seemingly good results.

## Description
This repository contains the implementation of:
* Recreate a camera design that performs class-specific all-optical imaging
* Algorithm for Classification of MNIST labels from all-optical erasure of undesired objects classes.
* U-net architecture for Image-To-Image Translation task, that reconstruct the all-optical erasure digits to the original MNIST digits

![project_diagram.png](project_diagram.png)


## Setup
### Requirements


Install required packages:


## Usage



### Data
we used MNIST dataset from torchvision Dataset class for training the Optical Model

### Models
* For implement the Optical Model, we used this article [To image, or not to image: Class-specific diffractive cameras with all-optical erasure of undesired objects](https://arxiv.org/abs/2205.13122).
* For implement the U-net we used exists [code](https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/unet.py) and change it for Image-To-Image task.



Big thanks to our supervisor Matan Kleiner who gave us the opportunity to undertake the project and supported us all along the way.

