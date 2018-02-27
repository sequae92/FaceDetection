## Modules for Loading Data

import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing

'''
Defining the psuedo code for the EM algorithm for a mixture of gaussian Models
The parameters unknown to us - 
1)the number of models for the given data,
2)the parameters of each of the models
3)the number of iterations

Consider the case where we have to fit 'n' Gaussian Models on a given dataset
The idea is to fit 'n' Gaussian models on a given data

Input: Training data , number of clusters
Output: Estimates of parameters, theta
'''

# Declaring all the global variables
pca = scaler = lold = 0
'''
This function will do all the global operations on the data, 'scaler' and 'pca' should be global variables
'''


def dataOperations(data):  # Make sure this RUNS after the data loading is done, so that scaler and pca objects are available
    # Calculating PCA, reducing the dimensionality
    global scaler
    global pca
    scaler = preprocessing.StandardScaler().fit(data)
    #scalerObj = returnScaler(data)
    Xfeatures = scaler.transform(data)
    # print(Xfeatures)
    # print(Xfeatures.shape)
    # print(Xfeatures)

    pca = PCA(n_components=50)
    # print(pca.explained_variance_ratio)
    Xnew = pca.fit_transform(Xfeatures)
    return Xnew  # returns the pca of the data

def returnScaled(d):
    #global scaler
    scaler = preprocessing.StandardScaler().fit(d)
    scaled = scaler.transform(d)
    return scaled

def loadData(flag):
    train_pos_path = '../dataset/train_pos/'
    train_neg_path = '../dataset/train_neg/'

    X1 = np.array([np.zeros(10800)])  # This is the feature vector for all positive images
    X0 = np.array([np.zeros(10800)])  # This is the feature vector for all negative images

    # First finding u1, c1 for all the positives
    for filename in os.listdir(train_pos_path):
        imgpath = train_pos_path + filename
        img = cv2.imread(imgpath, -1)
        # print(img.shape)
        x = np.asarray(img)
        x = np.reshape(x, (1, 10800))
        # print(x.shape)
        X1 = np.dstack([X1, x])
        # print(X1.shape)

    for filename in os.listdir(train_neg_path):
        imgpath = train_neg_path + filename
        img = cv2.imread(imgpath, -1)
        # print(img.shape)
        x = np.asarray(img)
        x = np.reshape(x, (1, 10800))
        # print(x.shape)
        X0 = np.dstack([X0, x])
    # print(X0.shape)

    Y1 = np.array([np.ones(1000)])
    # print(Y1.shape)

    Y0 = np.array([np.zeros(1000)])
    # print(Y0.shape)

    X1 = np.delete(X1, 0, axis=2)
    X1 = np.reshape(X1, (1000, 10800))
    # print(X1.shape)
    X0 = np.delete(X0, 0, axis=2)
    X0 = np.reshape(X0, (1000, 10800))
    # print(X0.shape)

    if flag == 1:
        return X1
    else:
        return X0