import numpy as np
import struct

imagefile = open("../data/train-images-idx3-ubyte", "rb")
labelfile = open("../data/train-labels-idx1-ubyte", "rb")

imagefile.read(16)
labelfile.read(8)

images = np.split(np.fromfile(imagefile, dtype="uint8"), 60000)
labels = np.split(np.fromfile(labelfile, dtype="uint8"), 60000)

sum6 = np.zeros(10)
sum13 = np.zeros(10)
counts = np.zeros(10)

for i in range(len(images)):
	sum6[labels[i]] += images[i][6*28 + 6]
	sum13[labels[i]] += images[i][13*28 + 13]
	counts[labels[i]] += 1

for i in range(0, 10):
	sum6[i] /= counts[i]
	sum13[i] /= counts[i]
	print "Average for " + str(i) + " for [6, 13]: [" + str(sum6[i]) + ", " + str(sum13[i]) + "]"
