#python script to resize the face images and making the training and testing set

import numpy as np
import cv2
import os
from os.path import join, exists
import random

DATA_PATH = './data'

def resize_image(img):
	resized_image = cv2.resize(img,(60,60))
	return resized_image

def resize_image_from_dir():
	#alldirnames = []
	path = './data'
	file = open('resizedimage.txt',"w")
	for d in os.listdir(path):
		dirpath = join(path,d)
		#print(dirpath)
		rdir = join(dirpath,'resized')
		#print(directory)
		if not exists(rdir):
			os.mkdir(rdir)
			print('Resized dir created!')
		else:	
			facedir = join(dirpath,'face')
			for filename in os.listdir(facedir):
				if filename.endswith(".jpg"):
					imgpath = join(facedir,filename)
					rimgpath = join(rdir,filename)
					img = cv2.imread(imgpath,1)
					resized_image = resize_image(img)
					cv2.imwrite(rimgpath,resized_image)
					file.write(rimgpath+'\n')

	file.close()

def make_train_and_test_set():
	file = open('resizedimage.txt',"r")
	data = list()
	for line in file:
		data.append(line)
	file.close()

	random.seed(1)
	random.shuffle(data)
	
	train_data = data[:int((len(data)+1)*.80)]
	test_data = data[int((len(data)+1)*.80+1):]

	train = np.random.choice(train_data,1000)
	test = np.random.choice(test_data,100)

	file.close()

	trainf = open('train1.txt',"w")
	testf = open('test1.txt',"w")

	for d in train:
		trainf.write(d)

	for d in test:
		testf.write(d)


'''
	path = './data'
	count = 0
	trainset = './train'
	testset = './test'

	if not exists(trainset):
		os.mkdir(trainset)
	if not exists(testset):
		os.mkdir(testset)

	for d in os.listdir(path):
		dirpath = join(path,d)
		rdir = join(dirpath,'resized')
		for file in os.listdir(rdir):
			if filename.endswith(".jpg"):
'''

if __name__ == '__main__':
	make_train_and_test_set()


