# Implement Autoencoder and K-means on Cifar-10 dataset
## Overview
### Implement autoencoders on cifar-10 dataset and generate sparse representation of images, Then displayed the generated images of autoencoder. Futher used the spatial 
representation of the images and implemented K-Means clustering on it with K-value as 10 and computed the Silhoutte score for 10 clusters.

## Dataset
### The dataset contains training set of 50000 examples and a test set of 10000 examples. Each example is 32x32 image where each image is 32 pixel in height and 32 pixel in width for a total of 1024 pixel in total. Each
image 32x32 matrix is reshaped into a 1024 size vector. The pixel value is an integer between 0 and 255. Each image belong to 10 classes namely: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship and Truck.

## Implementation
### First I have created the convolution autoencoder model where I have created encoder and decoder model seperately and then combined them both to create the final model. 
### Next I have used the sparse representation of encoder and implemented K-means with k=10 clusters to classify images using the sparse representation of the images.
