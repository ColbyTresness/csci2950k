import numpy as np
import struct
import math
import random

#open files
imagetrain = open("../data/train-images-idx3-ubyte", "rb")
labeltrain = open("../data/train-labels-idx1-ubyte", "rb")
imagetest= open("../data/t10k-images-idx3-ubyte", "rb")
labeltest = open("../data/t10k-labels-idx1-ubyte", "rb")

#read headers
imagetrain.read(16)
labeltrain.read(8)
imagetest.read(16)
labeltest.read(8)

#split into arrays
images = np.split(np.fromfile(imagetrain, dtype="uint8"), 60000)
labels = np.split(np.fromfile(labeltrain, dtype="uint8"), 60000)
timages = np.split(np.fromfile(imagetest, dtype="uint8"), 10000)
tlabels = np.split(np.fromfile(labeltest, dtype="uint8"), 10000)

#normalize images
images[:] = [x / 255.0 for x in images]
timages[:] = [x / 255.0 for x in timages]

w = np.zeros([784, 10])
b = np.zeros(10)

l = .5
numsteps = 10000

random.seed(88)


for x in range(numsteps):
	h = np.zeros(10)
	c = random.randint(0, len(images)-1)
	p = np.zeros(10)
	softmaxdenom = 0
	for j in range(10):
		for i in range(784):
			h[j] += w[i][j] * images[c][i]
		h[j] += b[j]
		softmaxdenom += math.exp(h[j])

	for j in range(10):
		p[j] = math.exp(h[j])/softmaxdenom


	E = -math.log(p[labels[c]])

	#backwards pass
	dEdh = np.zeros(10)
	dEdb = np.zeros(10)
	for j in range(10):
		if labels[c] == j:
			dEdh[j] = p[j] - 1
		else:
			dEdh[j] = p[j]
		dEdb[j] = dEdh[j]
		dEdw = np.zeros([784, 10])
		for i in range(784):
			dEdw[i][j] = images[c][i]*dEdh[j]
			w[i][j] -= l*dEdw[i][j]
		b[j] -= l*dEdb[j]

numcorrect = 0
for c in range(len(timages)):
	h = np.zeros(10)
	p = np.zeros(10)
	softmaxdenom = 0
	for j in range(10):
		for i in range(784):
			h[j] += w[i][j] * timages[c][i]
		h[j] += b[j]
		softmaxdenom += math.exp(h[j])

	maxIndex = 0
	maxProb = 0
	for j in range(10):
		p[j] = math.exp(h[j])/softmaxdenom
		if p[j] > maxProb:
			maxProb = p[j]
			maxIndex = j

	if tlabels[c][0] == maxIndex:
		numcorrect += 1

print float(numcorrect) / float(len(timages))
