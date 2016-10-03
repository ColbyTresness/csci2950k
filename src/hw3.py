from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '../data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_img = tf.reshape(x, [-1, 28, 28, 1])


#first conv layer
Wc1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
bc1 = tf.Variable(tf.constant(0.1, shape=[32]))


#second conv layer
Wc2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
bc2 = tf.Variable(tf.constant(0.1, shape=[64]))


#FF 1
Wf1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
bf1 = tf.Variable(tf.constant(0.1, shape=[1024]))

#FF 2
Wf2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
bf2 = tf.Variable(tf.constant(0.1, shape=[10]))

#conv and pool
hc1 = tf.nn.relu(tf.nn.conv2d(x_img, Wc1, strides=[1, 1, 1, 1], padding='SAME') + bc1)
hp1 = tf.nn.max_pool(hc1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#conv and pool
hc2 = tf.nn.relu(tf.nn.conv2d(hp1, Wc2, strides=[1, 1, 1, 1], padding='SAME') + bc2)
hp2 = tf.nn.max_pool(hc2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#reshape
hp2_flat = tf.reshape(hp2, [-1, 7*7*64])

#first feed forward
hf1 = tf.nn.relu(tf.matmul(hp2_flat, Wf1) + bf1)

#dropout
keep_prob = tf.placeholder(tf.float32)
hf1_drop = tf.nn.dropout(hf1, keep_prob)

#second feed forward
y = tf.nn.softmax(tf.matmul(hf1_drop, Wf2) + bf2)

# Define loss and optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
sess.run(tf.initialize_all_variables())


for i in range(2000):
	print(i)
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))