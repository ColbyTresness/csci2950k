import tensorflow as tf

sess = tf.InteractiveSession()

# Open files
trainFile = open("../data/train.txt", "rb")
testFile = open("../data/test.txt", "rb")

#lists of integers corresponding to the files
trainList = []
testList = []

# word->int dictionary
wordNums = {}

# populating wordnums based on training data
# also populates list of integers corresponding to training text
print("populating word->int dictionary")
numWords = 1
for line in trainFile:
	for word in line.split():
		if word not in wordNums:
			wordNums[word] = numWords
			numWords += 1
		trainList.append(wordNums[word])

# populates list of integers corresponding to test text
for line in testFile:
	for word in line.split():
		testList.append(wordNums[word])

# constants
EMBEDSIZE = 30
HIDDEN = 100
BATCHSIZE = 20
TRAININGRATE = 1e-4
EPOCHS = 1

print("creating model")
# Create the model
inpt = tf.placeholder(tf.int64, [None])
out = tf.placeholder(tf.int64, [None])
E = tf.Variable(tf.random_uniform([numWords, EMBEDSIZE], minval=-1, maxval=1, dtype=tf.float32, seed=0))
embed = tf.nn.embedding_lookup(E, inpt)

# set up forward pass
w1 = tf.Variable(tf.truncated_normal([EMBEDSIZE, HIDDEN], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[1, HIDDEN]))

h = tf.nn.relu(tf.matmul(embed, w1) + b1)

w2 = tf.Variable(tf.truncated_normal([HIDDEN, numWords], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[1, numWords]))

logits = tf.matmul(h, w2) + b2

# set up loss
error = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, out)

# set up perplexity node
perplexity = tf.exp(tf.reduce_mean(error))

# set up training step
train_step = tf.train.AdamOptimizer(TRAININGRATE).minimize(error)

# initialize
sess.run(tf.initialize_all_variables())

print("training")
# train
for e in range(EPOCHS):
	x = 0
	while x + BATCHSIZE < len(trainList):
		_, err = sess.run([train_step, error], feed_dict = {inpt: trainList[x : x + BATCHSIZE], out: trainList[x+1 : x+1+BATCHSIZE]})
		if x % 5000 == 0:
			print("perplexity for " + str(x) + ": %g"%perplexity.eval(feed_dict = {inpt: trainList[x : x + BATCHSIZE], out: trainList[x+1 : x+1+BATCHSIZE]}))
		x += BATCHSIZE


print("testing")
# test
err = sess.run(error, feed_dict = {inpt: trainList[0 : len(testList)-1], out: trainList[1 : len(testList)]})
print("test perplexity %g"%perplexity.eval(feed_dict = {inpt: testList[0 : len(testList)-1], out: testList[1 : len(testList)]}))

# 293.386