import os, sys, glob
import numpy as np
import cv2 as cv
import time

import helpers

def generate_model():

	paths = [
		'Empty',
		'Accent',
		'Accents/dynamic',
		'Accents/fermata',
		'Accents/Harmonic',
		'Accents/marcato',
		'Accents/mordent',
		'Accents/staccatissimo',
		'Accents/staccato',
		'Accents/stopped',
		'Accents/Tenuto',
		'Accents/turn',
		'AltoCleff',
		'BarLines',
		'BassClef',
		'Beams',
		'Breve',
		'Dots',
		'Flat',
		'Naturals',
		'NoteHeadsFlags/demisemiquaver',
		'NoteHeadsFlags/hemidemisemiquaver',
		'NoteHeadsFlags/quaver',
		'NoteHeadsFlags/semiquaver',
		'Notes',
		'NotesFlags',
		'NotesOpen',
		'Relation/Chord',
		'Relation/Glissando',
		'Relation/Slur-Tie',
		'Relation/Tuplet',
		'Relations',
		'Rests/demisemiquaver',
		'Rests/doublewhole',
		'Rests/half-whole',
		'Rests/hemidemisemiquaver',
		'Rests/quaver',
		'Rests/semiquaver',
		'Rests1',
		'Rests2',
		'SemiBreve',
		'Sharps',
		'TimeSignatureL',
		'TimeSignatureN',
		'TimeSignaturesL/CwithBar',
		'TimeSignaturesL/CwithoutBar',
		'TimeSignaturesN/0',
		'TimeSignaturesN/1',
		'TimeSignaturesN/2',
		'TimeSignaturesN/3',
		'TimeSignaturesN/4',
		'TimeSignaturesN/5',
		'TimeSignaturesN/6',
		'TimeSignaturesN/7',
		'TimeSignaturesN/8',
		'TimeSignaturesN/9',
		'TrebleClef'
	]

	n_data_per_path = []
	train_ims = []
	labels = []

	for i in range(len(paths)):

		n_data_found = 0

		for file_path in glob.glob(os.path.join('training_data', paths[i], "*")):
			#print(file_path)
			n_data_found = n_data_found+1
			labels.append(i)
			im = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
			resized = cv.resize(im, (20, 20))
			train_ims.append(np.array(resized, np.float32).reshape(-1))

		n_data_per_path.append(n_data_found)

	train_data = np.array(train_ims).astype(np.float32)
	print(train_data.shape)
	model_train_data = cv.ml.TrainData_create(train_data, cv.ml.ROW_SAMPLE, np.array(labels, np.int32))

	knn = cv.ml.KNearest_create()
	knn.train(model_train_data)

	return knn

def find_objs(image):

	knn = generate_model()

	im_pyramid = list(helpers.pyramid(image))
	for i in range(len(im_pyramid)):

		knn_data = []
		knn_data_pos = []

		for (x, y, window) in helpers.sliding_window(im_pyramid[i], stepSize=16, windowSize=(32, 64)):
			
			resized = cv.resize(window, (20, 20))
			knn_data.append(np.array(resized, np.float32).reshape(-1))
			knn_data_pos.append([x, y])
			
		nearest = knn.findNearest(np.array(knn_data, np.float32), k=5)
		clone = im_pyramid[i].copy()
		clone = cv.cvtColor(clone, cv.COLOR_GRAY2BGR)
		for j in range(len(nearest[1])):
			if nearest[1][j] == 24:
				cv.rectangle(clone, 
					(knn_data_pos[j][0], knn_data_pos[j][1]), 
					(knn_data_pos[j][0]+32, knn_data_pos[j][1]+64), 
					(0, 0, 255), 
					2)
		cv.imshow("Window", clone)
		cv.waitKey(0)