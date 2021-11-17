# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:15:12 2021

@author: harsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import cv2
import copy
from sklearn.metrics import silhouette_samples,silhouette_score

from sklearn.metrics import pairwise_distances
from validclust import dunn


import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


# Defining the K_means Class
class k_means():
    def __init__(self, clusters = 10, max_iters = 500):
        self.clusters = clusters
        self.max_iters = max_iters
        self.centroids = []
        self.loss_iters = []
        
    # Function to initialize the centroids randomly
    def initialize_centroids_rand(self, x_train):
        samples = x_train.shape[0]
        np.random.seed(np.random.randint(0, 10000))
        for i in range(self.clusters):
            random_index = np.random.choice(range(samples))
            self.centroids.append(x_train[random_index])

    # Function to initialize the centroids that are far away from each other
    def initialize_centroid_farthest(self, x_train):
        samples = x_train.shape[0]
        r = np.random.RandomState(63)
        rand_idx = r.randint(0, 10000)
        #rand_idx = np.random.randint(0, 10000)
        # Initializing first cluster randomly
        self.centroids.append(x_train[rand_idx])
        distances = []
        
        # Computing the remaining k-1 centroids that are far away from each other
        for l in range(self.clusters - 1):
            for i in range(samples):
                point = x_train[i]
                dist = float('inf')
                
                # Computing distance of this point from each centroid
                for j in range(len(self.centroids)):
                    temp_dist = np.linalg.norm(point - self.centroids[j])
                    dist = min(dist, temp_dist)
                distances.append(dist)
        
            # Choosing the point that has maximum distance as centroid
            distances = np.array(distances)
            next_c = x_train[np.argmax(distances)]
            self.centroids.append(next_c)
            distances = []
        
    # Function to initialize the clusters
    def initialize_clusters(self):
        self.clusters_dict = {'data':{i:[] for i in range(self.clusters)}}
    
    # Function to update the centroids after each iteration
    def update_centroids(self, x_train):
        for i in range(self.clusters):
            cluster = self.clusters_dict['data'][i]
            if(cluster == []):
                self.centroids[i] = x_train[np.random.choice(range(x_train.shape[0]))]
            else:
                self.centroids[i] = np.mean(np.vstack((self.centroids[i], cluster)), axis=0)
    
    # Function to calculate the loss
    def calculate_loss(self):
        loss = 0
        for k,v in list(self.clusters_dict['data'].items()):
            if v is not None:
                for v_ in v:
                    loss += np.linalg.norm(v_ - self.centroids[k])
        self.loss_iters.append(loss)
        return loss

    
    # Function to train the model
    def fit(self, x_train, y_train):
        self.y_pred = [None for _ in range(x_train.shape[0])]
        #self.initialize_centroids_rand(x_train)
        self.initialize_centroid_farthest(x_train)
        
        for i in range(self.max_iters):
            prev_centroids = copy.deepcopy(self.centroids)
            self.initialize_clusters()
            
            for j, sample in enumerate(x_train):
                dist = float('inf')
                
                # Calculating the distance of this point to each centroid and assigning the cluster likewise
                for k,centroid in enumerate(self.centroids):
                    dist_c = np.linalg.norm(sample - centroid)
                    if(dist_c < dist):
                        dist = dist_c
                        self.y_pred[j] = k
                if(self.y_pred[j] is not None):
                    self.clusters_dict['data'][self.y_pred[j]].append(sample)
                
            self.update_centroids(x_train)
            loss = self.calculate_loss()
            print("\n Iteration: ", i)
            print(np.linalg.norm(np.array(self.centroids) - np.array(prev_centroids)))
            if(np.linalg.norm(np.array(self.centroids) - np.array(prev_centroids)) < 5e-3):
                break

# Function to normalize the data        
def normalize(x):
    normalized_arr = (x - np.mean(x)) / np.var(x)
    return normalized_arr

# Function to extract the data and then normalize it.        
def get_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = normalize(np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_train]))
    x_test = normalize(np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_test]))
    return x_train, y_train, x_test, y_test


def run_kmeans():
    # Load the dataset
    x_train, y_train, x_test, y_test = get_data() 
    
    # Using the Test dataset and reshaping each 32x32 image to 1024 vector
    x_test2 = np.zeros((x_test.shape[0],1024))
    for i in range(x_test.shape[0]):
        x_test2[i] = x_test[i].reshape(-1)
            
    #Creating the Model
    k_means_model = k_means(clusters=10, max_iters=200)
    k_means_model.fit(x_test2, y_test) 
    
    # Calculating the Sillhoutte score
    print("\n\nSilhouette_score Score is : ")
    score = silhouette_score(x_test2, k_means_model.y_pred, metric='euclidean')
    print('%.3f' % score)
    
    # Calculating the  Dunn Index
    print("\n\nDunn Index is : ")
    dist = pairwise_distances(x_test2)
    index = dunn(dist, np.array(k_means_model.y_pred))
    print('%.3f '% index)

if __name__=="__main__":
    run_kmeans()
