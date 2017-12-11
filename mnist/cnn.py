import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

epochs = 25000
dstep = 100
plotSamples = 1000


plotSamples = min(epochs, plotSamples)
plotStep = int((epochs - 1)/(plotSamples - 1))

if not os.path.exists("./data/cnn"):
	os.makedirs("./data/cnn")

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
  
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
prediction = tf.argmax(y_conv, 1)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

trainAccuracyArray = np.zeros(plotSamples)
testAccuracyArray = np.zeros(plotSamples)
epochArray = np.zeros(plotSamples)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	t0 = time.time()
	for e in range(epochs):
		batch = mnist.train.next_batch(50)
		if(e%dstep == 0):
			trainAccuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
			testAccuracy =  sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
			print('{:5d}: {:7.7f}, {:7.7f}'.format(e, trainAccuracy, testAccuracy))
		# if(e%plotStep == 0):
		# 	i = e//plotStep
		# 	epochArray[i] = e
		# 	trainAccuracyArray[i] = sess.run(accuracy, feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
		# 	testAccuracyArray[i] = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
		sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.3})
	print("Duration: {}".format(time.time() - t0))
	print('test accuracy: {:7.7f}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
	np.save("./data/cnn/error", [epochArray, trainAccuracyArray, testAccuracyArray])
	
	predicted = sess.run(prediction, feed_dict = {x: mnist.test.images, keep_prob: 1.0})
	labels = [np.argmax(l) for l in mnist.test.labels]
	
	wrongLabel = []
	wrongPrediction = []
	wrongIndex = []
	
	for i in range(len(labels)):
		if(labels[i] != predicted[i]):
			wrongIndex.append(i)
			wrongLabel.append(labels[i])
			wrongPrediction.append(predicted[i])
			print("[{}, {}, {}],\\".format(i, labels[i], predicted[i]))