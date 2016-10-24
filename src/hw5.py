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
	if counter < 8000:
		wordIndices[word] = counter
	else:
		wordIndices[word] = 8000
	counter += 1

trainWords = list()
testWords = list()

vocabSize = maxVocabSize + 1
i = 0
for word in vocab:
	i += 1
	if i < math.floor(len(vocab)*.9):
		trainWords.append(wordIndices[word])
	else:
		testWords.append(wordIndices[word])

trains = np.array(trainWords)
tests = np.array(testWords)


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

basicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell(LSTMSIZE, state_is_tuple=True)
initialState = basicLSTMCell.zero_state(BATCHSIZE, tf.float32)

w = tf.Variable(tf.truncated_normal([LSTMSIZE, vocabSize], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[vocabSize]))

embedDrop = tf.nn.dropout(embed, keep_prob)

rnn, outst = dyrnn = tf.nn.dynamic_rnn(basicLSTMCell, embedDrop, initial_state = initialState)

rnn2 = tf.reshape(rnn, [BATCHSIZE * NUMSTEPS, LSTMSIZE])

logits = tf.matmul(rnn2, w) + b

wgts = tf.Variable(tf.constant(1.0, shape=[BATCHSIZE*NUMSTEPS]))
y1d = tf.reshape(y, [BATCHSIZE*NUMSTEPS])

loss1 = tf.nn.seq2seq.sequence_loss_by_example([logits], [y1d], [wgts])
loss = tf.reduce_sum(loss1)

# set up training step
trainStep = tf.train.AdamOptimizer(TRAININGRATE).minimize(loss)

# initialize
sess.run(tf.initialize_all_variables())

print("training")
# train
lsum = 0
iters = 0
for e in range(EPOCHS):
	i = 0
	state = (np.zeros([BATCHSIZE, LSTMSIZE]), np.zeros([BATCHSIZE, LSTMSIZE]))
	while i + BATCHSIZE*NUMSTEPS + 1 < len(trains):
		iters += 1
		nextstate, _, l = sess.run([outst, trainStep, loss], 
			feed_dict = {x: np.reshape(trains[i : i + BATCHSIZE*NUMSTEPS], (BATCHSIZE, NUMSTEPS)),
						 y: np.reshape(trains[i+1 : i+1+BATCHSIZE*NUMSTEPS], (BATCHSIZE, NUMSTEPS)),  
						 keep_prob: .5,
						 initialState: state})
		print(math.exp(l/(BATCHSIZE*NUMSTEPS)))
		i += BATCHSIZE*NUMSTEPS
		lsum += (l / (BATCHSIZE*NUMSTEPS))
		state = nextstate
print("train perplexity: ")
print(math.exp(lsum/iters))
print("lsum: ")
print(lsum)
print("iters: ")
print(iters)
print("lsum/iters: ")
print(lsum/iters)


print("testing")
# test
iters = 0
lsum = 0
for e in range(EPOCHS):
	i = 0
	state = (np.zeros([BATCHSIZE, LSTMSIZE]), np.zeros([BATCHSIZE, LSTMSIZE]))
	while i + BATCHSIZE*NUMSTEPS + 1 < len(tests):
		iters += 1
		nextstate, _, l = sess.run([outst, trainStep, loss], 
			feed_dict = {x: np.reshape(tests[i : i + BATCHSIZE*NUMSTEPS], (BATCHSIZE, NUMSTEPS)),
						 y: np.reshape(tests[i+1 : i+1+BATCHSIZE*NUMSTEPS], (BATCHSIZE, NUMSTEPS)),
						 keep_prob: 1.0, 
						 initialState: state})
		print(math.exp(l/(BATCHSIZE*NUMSTEPS)))
		i += BATCHSIZE*NUMSTEPS
		state = nextstate
		lsum += (l / (BATCHSIZE*NUMSTEPS))
print("test perplexity: ")
print(math.exp(lsum/iters))

