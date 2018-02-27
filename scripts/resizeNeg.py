# resize the images in the negative

import os
import cv2
import data_prep

path = ['../dataset/test_neg/','../dataset/train_neg/']
def resizeImages():
	for p in path:
		for filename in os.listdir(p):
			imgpath = p + filename
			print(imgpath)
			img = cv2.imread(imgpath)
			print(img.shape)
			crop_img = img[0:60,0:60]
			cv2.imwrite(imgpath,crop_img)

def main():
	resizeImages()

if __name__ == '__main__':
	main()
