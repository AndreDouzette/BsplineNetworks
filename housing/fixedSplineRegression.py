import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from splineLayer import importSpline, splineParameters
import seaborn as sns
from housingData import housing
sns.set_style("ticks")
sns.set_context("poster")
trainInputs, trainTargets, testInputs, testTargets = housing()
spline = importSpline("../splineOp", True)

def randomVariable(size, std = 0.1, name = None):
	return tf.Variable(tf.random_normal(size, stddev = std), name = name)

epochs = 50000
dstep = 10000

splineSize = 21
inputSize = 8
hiddenSize = 5
outputSize = 1

try:
	p = int(sys.argv[1])
except:
	p = 3

samples = 15
for sample in range(samples):
	
	#Neural network architecture
	x = tf.placeholder(tf.float32, [None, inputSize])
	t = tf.placeholder(tf.float32, [None, 1])

	W1 = randomVariable([inputSize, hiddenSize])
	b1 = randomVariable([hiddenSize])
	W2 = randomVariable([hiddenSize, outputSize])
	b2 = randomVariable([outputSize])

	t1, c1 = splineParameters(hiddenSize, splineSize, p, [-2, 2])
	t2, c2 = splineParameters(outputSize, splineSize, p, [-2, 2])

	h = spline(tf.matmul(x, W1) + b1, t1, c1, p)
	y = spline(tf.matmul(h, W2) + b2, t2, c2, p)
	# y = tf.matmul(h, W2) + b2

	cost = tf.sqrt(tf.losses.mean_squared_error(t, y))
	_, var = tf.nn.moments(t, axes = [0])
	nrmse = tf.sqrt(tf.losses.mean_squared_error(t, y)/var[0])
	optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

	minError = 1e6

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for e in range(epochs):
			sess.run(optimizer, feed_dict = {x: trainInputs, t: trainTargets})
			tmpErr = sess.run(nrmse, feed_dict = {x: testInputs, t: testTargets})
			if(tmpErr < minError):
				minError = tmpErr
			if(e%dstep == 0):
				print("{:5d}: {}".format(e, sess.run(nrmse, feed_dict = {x: testInputs, t: testTargets})))
		t1s, t2s, c1s, c2s = sess.run([t1, t2, c1, c2])
		np.save("./splineData{}".format(p), [t1s, t2s, c1s, c2s])
		print("Min error(p = {:1d}): {}".format(p, minError))