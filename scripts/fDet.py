# Python code for Project 1 - Face Detection

from os import walk
import os
import random
import numpy as np
from random import randint
import urllib.request
import cv2


def data_prep(mypath):
	dirs = []
	files = []
	jpgFiles = []
	cnt = 0;

	if(os.path.isdir(mypath)):
		for(dirpath,dirnames,filenames) in walk(mypath):
			for dir in dirnames:
				#print os.path.join(dirpath,dir)
				for filename in os.listdir(os.path.join(dirpath,dir)):
					if(cnt < 1000):
						if filename.endswith('.jpg'):
							print(os.path.join(dir,filename))
							cnt = cnt+1
							files.append(os.path.join(dir,filename))	
	#print files
	print(len(files))
'''
This is to read the annotations file for the bounding box coordinates

def readAnnotations():

This is to read image from the URL listed in the info.txt
'''
def readImagefromURL(url):
	url_response = urllib.request.urlopen(url)
	img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
	img = cv2.imdecode(img_array, -1)

	#return img
	cv2.imshow('URL Image', img)
	cv2.waitKey()

#make training set
#this section will derive 1000 images from the folder
'''
	num = randint(1,len(f))
	print num
'''
def main():
	#mypath stores the data from the MSRA samples folder
	mypath = "C:/Users/spond/Desktop/ECE 763/Project/Project 1/MSRA/thumbnails_features_deduped_sample"
	#data_prep(mypath)
	url = "http://s0.geograph.org.uk/photos/40/57/405725_b17937da.jpg"
	readImagefromURL(url)

if __name__ == "__main__":
	main()	
