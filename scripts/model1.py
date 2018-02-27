#To implement the gaussian model over the training data

import os
import cv2
import numpy as np
import random

train_pos_path = '../dataset/train_pos'
test_pos_path = '../dataset/test_pos'
train_neg_path = '../dataset/train_neg'
test_neg_path = '../dataset/test_neg'

#First finding u1, c1 for all the positives
for filename in os.listdir(train_pos_path):
	imgpath = train_pos_path + filename
	img = cv2.imread(imgpath,1)
	x = np.asarray(img)


