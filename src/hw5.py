import tensorflow as tf
from collections import Counter
import re
import math
import numpy as np

sess = tf.InteractiveSession()

BOOKPATH = "../data/crimeandpunishment.txt"
maxVocabSize = 8000


def basic_tokenizer(sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
    """
    Very basic tokenizer: split the sentence into a list of tokens, lowercase.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(word_split, space_separated_fragment))
    return [w.lower() for w in words if w]

# Open file
book = open(BOOKPATH, "rb")

vocab = basic_tokenizer(book.read())

vocabCounts = Counter(vocab)

wordIndices = {}

counter = 1
for word in vocabCounts:
	if counter < 8001:
		wordIndices[word] = counter
	else:
		wordIndices[word] = 8001
	counter += 1

trainWords = []
testWords = []

vocabSize = len(vocab)
for i in range(vocabSize):
	if i < math.floor(vocabSize*.9):
		trainWords += vocab[i]
	else:
		testWords += vocab[i]

# constants
EMBEDSIZE = 50
BATCHSIZE = 50
TRAININGRATE = 1e-4
NUMSTEPS = 20
LSTMSIZE = 256
EPOCHS = 1

print("creating model")
# Create the model
x = tf.placeholder(tf.int32, [BATCHSIZE, NUMSTEPS])
y = tf.placeholder(tf.int32, [BATCHSIZE, NUMSTEPS])
keep_prob = tf.placeholder(tf.float32)
E = tf.Variable(tf.random_uniform([vocabSize, EMBEDSIZE], minval=-1, maxval=1, dtype=tf.float32, seed=0))
embed = tf.nn.embedding_lookup(E, x)

basicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell(LSTMSIZE)

initialState = basicLSTMCell.zero_state(BATCHSIZE, dtype=tf.float32)
w = tf.Variable(tf.truncated_normal([LSTMSIZE, vocabSize], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[1, vocabSize]))

embedDrop = tf.nn.dropout(embed, keep_prob)

rnn, outst = dyrnn = tf.nn.dynamic_rnn(basicLSTMCell, embedDrop, initial_state = initialState)

rnn2 = tf.reshape(rnn, [BATCHSIZE * NUMSTEPS, LSTMSIZE])

logits = tf.matmul(rnn2, w) + b

wgts = tf.Variable(tf.constant(1.0, shape=[1, BATCHSIZE*NUMSTEPS]))
y = tf.reshape(y, [1, BATCHSIZE*NUMSTEPS])
loss1 = tf.nn.seq2seq.sequence_loss_by_example([logits], [y], [wgts])
loss = tf.reduce_sum(loss1)
perplexity = tf.exp(loss / BATCHSIZE)

# set up training step
train_step = tf.train.AdamOptimizer(TRAININGRATE).minimize(perplexity)

# initialize
sess.run(tf.initialize_all_variables())

print("training")
# train
for e in range(EPOCHS):
	x = 0
	state = initialState.eval()
	while x + BATCHSIZE*NUMSTEPS + 1 < len(trainWords):
		print("x: " + str(x))
		print(type(state))
		print(state)
		nextstate, perp = sess.run([outst, perplexity], 
			feed_dict = {x: trainWords[x : x + BATCHSIZE*NUMSTEPS],
						 y: trainWords[x+1 : x+1+BATCHSIZE*NUMSTEPS], 
						 keep_prob: .5, 
						 initialState: state})
		if x % 5000 == 0:
			print("perplexity for " + str(x) + ": %g"%perplexity.eval(feed_dict = {x: trainWords[x : x + BATCHSIZE*NUMSTEPS], 
																				y: trainWords[x+1 : x+1+BATCHSIZE*NUMSTEPS], 
																				keep_prob: .5, 
						 														initialState: state}))
		x += BATCHSIZE
		state = nextstate


print("testing")
# test
for e in range(EPOCHS):
	x = 0
	state = (initialState[0].eval(), initialState[1].eval())
	while x + BATCHSIZE*NUMSTEPS + 1 < len(trainWords):
		nextstate, perp = sess.run([outst, perplexity], 
			feed_dict = {x: testWords[x : x + BATCHSIZE*NUMSTEPS],
						 y: testWords[x+1 : x+1+BATCHSIZE*NUMSTEPS], 
						 keep_prob: 1.0, 
						 initialState: state})
		if x % 5000 == 0:
			print("perplexity for " + str(x) + ": %g"%perplexity.eval(feed_dict = {x: testWords[x : x + BATCHSIZE*NUMSTEPS], 
																				y: testWords[x+1 : x+1+BATCHSIZE*NUMSTEPS], 
																				keep_prob: .5,
						 														initialState: state}))
		x += BATCHSIZE
		state = nextstate




# err = sess.run(error, feed_dict = {x: trainWords[0 : len(testWords)-1], y: trainWords[1 : len(testWords)], keep_prob: 1.0, initialState[0]: state[0], initialState[1]: initial_state[1]})
# print("test perplexity %g"%perplexity.eval(feed_dict = {x: testWords[0 : len(testWords)-1], y: testWords[1 : len(testWords)], keep_prob: 1.0, initialState[0]: initial_state[0], initialState[1]: initial_state[1]}))

# 293.386