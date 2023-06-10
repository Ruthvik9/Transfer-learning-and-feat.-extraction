# Transfer-learning-and-feat.-extraction
![image](https://github.com/Ruthvik9/Transfer-learning-and-feat.-extraction/assets/74010232/ca60b153-3835-4328-a923-f71c9b292564)


I've also created a simple site to visualize all the plots I've generated with the extracted features - https://pipesextractedimgplot.netlify.app/

## VGG16 Feature Extraction for Custom Datasets
This repository presents a method for feature extraction from custom datasets using the VGG16 neural network pre-trained on the ImageNet dataset. Features extracted in this way can serve as input for a wide variety of downstream tasks, such as transfer learning, classification, clustering, visualization, and more.

## Overview
The repository includes Python code that takes advantage of hooks in PyTorch to extract feature vectors from the VGG16 network. The process involves loading an image dataset, passing it through the network, and collecting the feature vectors produced by a specific layer in the network.

## Dataset
The code is designed to be used with any custom dataset. As an example, we use the pacp dataset, containing images of defective pipes. This dataset is read and processed using a custom pacp class, which extends PyTorch's Dataset class.

## Feature Extraction
The features are extracted from the VGG16 model's 'avgpool' layer using PyTorch hooks. PyTorch hooks are a powerful tool that allow us to extract intermediate values within the network, such as the outputs of individual layers during the forward pass or backward pass. In this code, a forward hook is registered to the 'avgpool' layer of the VGG16 model to collect the output feature maps. This method allows us to save the exact outputs of this layer when an image is passed through the network.

## Result
The feature vectors obtained are then pickled for future use, e.g., plotting distributions of datapoints after applying PCA, or as inputs to another machine learning model. Although the PCA visualization code is not included in this repository, the pickled data can be easily loaded and used in any such application.

## Usage
The code can be run as a script on the command line, and it expects the file names of the images to be stored in a pickle file. The images are assumed to be in .jpg format, and their directory path must be specified when initializing the pacp dataset. The DataLoader parameters can be adjusted based on the resources available.
