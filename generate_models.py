import os, sys, glob
import numpy as np
import cv2 as cv

paths = [
	'notes',
	'unknown'
]

n_data_per_path = []
train_data = []
labels = []

for i in range(len(paths)):

	n_data_found = 0

	for file_path in glob.glob(os.path.join('training_data', paths[i], "*")):
		#print(file_path)
		n_data_found = n_data_found+1
		labels.append(i)
		im = cv.imread(file_path)
		gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
		train_data.append(np.array(gray))

	n_data_per_path.append(n_data_found)

train_data = np.array(train_data)

knn = cv.ml.KNearest_create()
knn.train(train_data, cv.ml.ROW_SAMPLE, labels)