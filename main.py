import os, sys, numpy
import cv2 as cv

def check_img(img):
	imgS = cv.resize(img, (int(len(img[0])/2), int(len(img)/2)))
	cv.imshow("Image", imgS);
	cv.waitKey(0)

if len(sys.argv) < 2:
	print("Please enter the image path as an argument.")
	sys.exit(0)

imgpath = sys.argv[1]
if not os.path.exists(imgpath):
	print("Invalid image path.")
	sys.exit(0)

img = cv.imread(imgpath, cv.IMREAD_GRAYSCALE)

# binarização

th, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# detecção de pautas

img_bin = numpy.float32(img_bin)
hist = cv.reduce(img_bin, 1, cv.REDUCE_SUM)

staff_range = min(hist)*1.1 # numero magico... arrumar depois
staff_rows = []
for i in range(len(hist)):
	if hist[i] < staff_range:
		staff_rows.append(i)

# agrupando valores da mesma pauta:

grouped_staff_rows = []
l = []

for n in staff_rows:

	if l == [] or n == l[-1] + 1:
		l.append(n)
	else:
		grouped_staff_rows.append(l)
		l = []
		l.append(n)

grouped_staff_rows.append(l)

for line in grouped_staff_rows:
	print(f"Staff row found at: {line[0]}-{line[-1]}")

# remoção de pautas:

for i in staff_rows:
	for j in range(0, len(img[0])):
		if img[i-1][j] == 255:
			img[i][j] = 255

check_img(img)
