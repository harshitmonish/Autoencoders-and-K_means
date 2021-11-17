# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:13:52 2021

@author: harsh
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import cv2
import copy
from sklearn.metrics import silhouette_samples,silhouette_score
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose
from sklearn.metrics import pairwise_distances
from validclust import dunn
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score

from sklearn.metrics import pairwise_distances
from validclust import dunn

# Fetching the Dataset
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
    

# Class to create my autoencoder model
class autoencoders():
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.outputs = [] 
        
    # This function builds my autoencoder model. 
    """
    The model has following structure :
    Encoder:
        1 Conv layer
        1 MaxPooling layer
        1 BatchNormalization Layer
        1 Conv layer
        1 MaxPooling layer
        1 BatchNormalization Layer
        1 Conv Layer (creates the 8x8x3 image sparse representation)
        
    Decoder:
        1 Conv Layer
        1 BatchNormalization layer
        1 Conv Layer
        1 BatchNormalization layer
        1 conv Layer (Output Layer).
    """
    def build_model(self):
        #build encoder
        ip = layers.Input(shape=(32, 32, 3))
        e = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3))(ip)
        e = MaxPooling2D(2, 2, padding='same')(e)
        e = BatchNormalization()(e)
        e = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(e)
        e = MaxPooling2D(2, 2, padding='same')(e)
        e = BatchNormalization()(e)
        e_o = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='relu')(e)
        
        #build decoder
        #d = UpSampling2D()(e_o)
        d = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(e_o)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(d)
        d = BatchNormalization()(d)
        d_o = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid')(d)
        
        self.model = Model(inputs = ip, outputs=d_o)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()   
        self.encoder = Model(inputs = ip, outputs = e_o)
        
    # Function to train the model 
    def fit(self, xtrain, ytrain, xtest, ytest):
        self.build_model()
        print(" \n Training the Autoencoder")
        for i in range(self.epochs):
            #noise = np.random.normal(0, NOISE, xtrain.shape)
            history = self.model.fit(xtrain, xtrain, epochs=30, batch_size=1000, validation_data =(xtest, xtest))
        return history
    
    # Function to get the output of the encoder which is referred to embedings or sparse representations      
    def get_embed(self, xtrain):
        print("\n Fetching the Encoder output, the sparse representation of the images")
        train_embed = self.encoder.predict(xtrain)
        return train_embed
    
    # Function to get the original images back from the autoencoder.
    def predict(self, xtest):
        print("\n Generating the images from autoencoder model")
        return self.model.predict(xtest)

# Function to apply KMeans on encoder sparse representation output.
def k_means_compute(embed):
    print("\n Implementing K means on sparse represenation of Encoder")
    kmeans = KMeans(init="k-means++", n_clusters=10, random_state=0)
    embed_trans = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in embed])
    embed_trans2 = np.zeros((embed_trans.shape[0],64))
    for i in range(embed_trans.shape[0]):
        embed_trans2[i] = embed_trans[i].reshape(-1)
    kmeans.fit(embed_trans2)
    score = silhouette_score(embed_trans2, kmeans.labels_, metric='euclidean')
    print("\n Silhouette_score Score is : ")
    print('%.3f' % score)
    
    #print("DB Index is : ")
    #dist = pairwise_distances(embed_trans2)
    #index = dunn(dist, np.array(kmeans.labels_))
    #print('%.3f '% index)

# Function to normalize the input image            
def normalize(x):
    normalized_arr = x/255.0
    return normalized_arr

# Function to get the input data and normalize it.        
def get_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = normalize(np.array(x_train))
    x_test = normalize(np.array(x_test))
    return x_train, y_train, x_test, y_test

# Function to plot the train and validation loss graph
def plot_train_valid_loss(his):
    print("\n Displaying the Plot of train vs validation loss")
    matplotlib.interactive(True)
    fig1 =  plt.figure(figsize=(15, 10))
    plt.plot(his.history['loss'], label='train')
    plt.plot(his.history['val_loss'], label='test')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper right')
    plt.show()    
    fig1.savefig("autoencoder_train_valid_loss.png")

def display_images(img1, img2):
    print("\n Displaying the input and output images of the autoencoder")
    matplotlib.interactive(True)
    fig2 =  plt.figure(figsize=(60, 60))
    plt.imshow(sbscompare(img1 ,img2, 20, 20))
    plt.axis('off')
    plt.show()
    fig2.savefig("autoencoder_images_comparision.png")
    
    

#Function to display the images by stacking them side by side
def hstackimgs(min, max, img):
    return np.hstack(img[i] for i in range(min, max))

def sqstackimgs(l, h, img):
    return np.vstack(hstackimgs(i*l, (i+1)*l, img) for i in range(h)) 

def sbscompare(img1, img2, l, h):
    A = sqstackimgs(l, h, img1)
    B = sqstackimgs(l, h, img2)
    C = np.ones((A.shape[0], 32, 3))
    return np.hstack((A, B, C))

# Main function
def run_autoencoder():
    # Getting the data
    x_train, y_train, x_test, y_test = get_data() 
    
    # Creating the Autoencoder model
    auto_model = autoencoders(0.001, 1)
    
    # Training the autoencoder model
    his = auto_model.fit(x_train, y_train, x_test, y_test)
    
    #Plotting the train vs validation loss. test data is taken as validation data
    plot_train_valid_loss(his)
    
    # Getting the output of the encoder 
    embed = auto_model.get_embed(x_train)
    
    # Kmeans implemented on sparse representation of encoder
    k_means_compute(embed)
    
    # Getting the output images of the Autoencoder model
    pred_img = auto_model.predict(x_train)
    
    # Displaying the train and predicted images.
    display_images(x_train, pred_img)
    

if __name__=="__main__":
    run_autoencoder()