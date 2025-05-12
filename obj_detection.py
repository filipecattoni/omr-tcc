import os, sys, glob, time
from enum import Enum
import numpy as np
import cv2 as cv

import helpers

class Symbols(Enum):
	empty                = 0
	unknown              = 1
	note_whole           = 2
	note_half_upright    = 3
	note_half_flipped    = 4
	note_quarter_upright = 5
	note_quarter_flipped = 6
	rest_whole_half      = 7
	rest_quarter         = 8
	rest_eighth          = 9
	rest_sixteenth       = 10
	rest_thirtysecond    = 11
	accent_accented      = 12
	accent_marcato       = 13
	accent_staccato      = 14
	accidental_flat      = 15
	accidental_natural   = 16
	accidental_sharp     = 17
	clef_alto            = 18
	clef_bass            = 19
	clef_treble          = 20
	timesig_common       = 21
	timesig_cut          = 22

paths = [
	'empty',
	'unknown',
	'notes/whole',
	'notes/half_upright',
	'notes/half_flipped',
	'notes/quarter_upright',
	'notes/quarter_flipped',
	'rests/whole-half',
	'rests/quarter',
	'rests/eighth',
	'rests/sixteenth',
	'rests/thirtysecond',
	'accents/accented',
	'accents/marcato',
	'accents/staccato',
	'accidentals/flat',
	'accidentals/natural',
	'accidentals/sharp',
	'clefs/alto',
	'clefs/bass',
	'clefs/treble',
	'time_sigs/common',
	'time_sigs/cut'
]

def generate_model():

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
	knn.save('model.yml')

	return knn

def find_objs(image):

	knn = generate_model()

	im_pyramid = list(helpers.pyramid(image))
	for i in range(len(im_pyramid)):

		knn_data = []
		knn_data_pos = []

		for (x, y, window) in helpers.sliding_window(im_pyramid[i], stepSize=8, windowSize=(40, 80)):
			
			resized = cv.resize(window, (20, 20))
			knn_data.append(np.array(resized, np.float32).reshape(-1))
			knn_data_pos.append([x, y])
			
		nearest = knn.findNearest(np.array(knn_data, np.float32), k=1)
		clone = im_pyramid[i].copy()
		clone = cv.cvtColor(clone, cv.COLOR_GRAY2BGR)
		for j in range(len(nearest[1])):
			if Symbols.note_quarter_flipped.value in nearest[2][j]:
				cv.rectangle(clone, 
					(knn_data_pos[j][0], knn_data_pos[j][1]), 
					(knn_data_pos[j][0]+40, knn_data_pos[j][1]+80), 
					(0, 0, 255), 
					2)
		cv.imshow("Window", clone)
		cv.waitKey(0)