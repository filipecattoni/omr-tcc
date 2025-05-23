# scripts de fontes externas utilizadas para auxiliar o desenvolvimento.
import cv2 as cv
import imutils

# retorna uma piramide de imagens para fazer deslizamento de janela.
# fonte: https://pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
def pyramid(image, scale=1.5, minSize=(50, 50)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image

# retorna uma lista de janelas da imagem de input 
# fonte: https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])