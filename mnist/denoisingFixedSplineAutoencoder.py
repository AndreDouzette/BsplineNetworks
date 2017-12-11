import sys
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from splineLayer import importSpline, splineParameters
import seaborn as sns
sns.set_style("ticks")
sns.set_context("poster")
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
spline = importSpline("../splineOp", True)

epochs = 25000
encoderEpochs = 15000
batchSize = 256

plotSamples = 1000

plotSamples = min(epochs, plotSamples)
plotStep = int((epochs - 1)/(plotSamples - 1))

displayStep = 100
exhibitionSize = 3

inputSize = 28*28 #784
hidden1 = 200
hidden2 = 50
outSize = 10

try:
	internalKnots = int(sys.argv[1])
except:
	internalKnots = 5
try:
	p = int(sys.argv[2])
except:
	p = 3

runNumber = len(glob.glob("./data/fixedSplineDAE{}-{}*".format(internalKnots, p)))
print(runNumber)

if not os.path.exists("./data/fixedSplineDAE{}-{}-{}".format(internalKnots, p, runNumber)):
	os.makedirs("./data/fixedSplineDAE{}-{}-{}".format(internalKnots, p, runNumber))

splineSpaceSize = internalKnots + p + 1

def randomVariable(size, std = 0.1, name = ""):
	return tf.Variable(tf.random_normal(size, stddev = std), name = name)

def corrupt(x):
	return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32), tf.float32))

learningRateAE = 1e-3
learningRateFT = 1e-4
lossFunction = tf.losses.mean_squared_error
displayImages = mnist.validation.images[[3, 5, 7, 9, 0, 66, 71, 85, 4]]

x = tf.placeholder(tf.float32, [None, inputSize])
target = tf.placeholder(tf.float32, [None, outSize])

t1, C1 = splineParameters(hidden1, splineSpaceSize, p,   [-5, 5], name = "spline1")
t2, C2 = splineParameters(hidden2, splineSpaceSize, p,   [-5, 5], name = "spline2")
t3, C3 = splineParameters(outSize, splineSpaceSize, p,   [-5, 5], name = "spline3")
_, C1T = splineParameters(inputSize, splineSpaceSize, p, [-5, 5], name = "spline1T")
_, C2T = splineParameters(hidden1, splineSpaceSize, p,   [-5, 5], name = "spline2T")

W1 = randomVariable([inputSize, hidden1], name = "w1")
b1 = randomVariable([hidden1], name = "b1")
W2 = randomVariable([hidden1, hidden2], name = "w2")
b2 = randomVariable([hidden2], name = "b2")
W3 = randomVariable([hidden2, outSize], name = "w3")
b3 = randomVariable([outSize], name = "b4")

W1T = tf.matrix_transpose(W1)
W2T = tf.matrix_transpose(W2)
b1T = randomVariable([inputSize], name = "b1T")
b2T = randomVariable([hidden1], name = "b2T")

target1 = x
target2 = spline(tf.matmul(x, W1) + b1, t1, C1, p)
encoder1 = spline(tf.matmul(corrupt(x), W1) + b1, t1, C1, p)
encoder2 = spline(tf.matmul(corrupt(target2), W2) + b2, t2, C2, p)
decoder1 = spline(tf.matmul(encoder1, W1T) + b1T, t1, C1T, p)
decoder2 = spline(tf.matmul(encoder2, W2T) + b2T, t2, C2T, p)

kp = tf.placeholder(tf.float32)
y = spline(tf.matmul(target2, W2) + b2, t2, C2, p)
# y = spline(tf.matmul(y, W3) + b3, t3, C3, p)
y = tf.matmul(tf.nn.dropout(y, kp), W3) + b3

loss1 = lossFunction(target1, decoder1)
loss2 = lossFunction(target2, decoder2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=y))
# cost = tf.reduce_mean((tf.nn.softmax(y) - target)**2)

optimizer1 = tf.train.AdamOptimizer(learningRateAE).minimize(loss1, var_list = [W1, b1, t1, C1, b1T, C1T])
optimizer2 = tf.train.AdamOptimizer(learningRateAE).minimize(loss2, var_list = [W2, b2, t2, C2, b2T, C2T])
optimizer3 = tf.train.AdamOptimizer(learningRateFT).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(target, 1)), tf.float32))
trainAccuracyArray = np.zeros(plotSamples)
testAccuracyArray = np.zeros(plotSamples)
epochArray = np.zeros(plotSamples)

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	print("\nFirst autoencoding layer")
	for e in range(encoderEpochs):
		images, _ = mnist.train.next_batch(batchSize)
		sess.run(optimizer1, feed_dict = {x: images})
		if(e%displayStep == 0):
				loss = sess.run(loss1, feed_dict = {x: mnist.test.images})
				print("{:5d}: {:7.7f}".format(e, loss))
	saver.save(sess, "./data/fixedSplineDAE{}-{}-{}/firstLayer.ckpt".format(internalKnots, p, runNumber))
	
	print("\nSecond autoencoding layer")
	for e in range(encoderEpochs):
		images, _ = mnist.train.next_batch(batchSize)
		sess.run(optimizer2, feed_dict = {x: images})
		if(e%displayStep == 0):
				loss = sess.run(loss2, feed_dict = {x: mnist.test.images})
				print("{:5d}: {:7.7f}".format(e, loss))
	saver.save(sess, "./data/fixedSplineDAE{}-{}-{}/secondLayer.ckpt".format(internalKnots, p, runNumber))
	
	print("\nFine tuning and last layer")
	for e in range(epochs):
			images, labels = mnist.train.next_batch(batchSize)
			sess.run(optimizer3, feed_dict = {x: images, target: labels, kp: 0.5})
			if(e%plotStep == 0):
				i = e//plotStep
				epochArray[i] = e
				trainAccuracyArray[i] = sess.run(accuracy, feed_dict = {x: mnist.train.images, target: mnist.train.labels, kp: 1.0})
				testAccuracyArray[i] = sess.run(accuracy, feed_dict = {x: mnist.test.images, target: mnist.test.labels, kp: 1.0})
			if(e%displayStep == 0):
				trainAccuracy = sess.run(accuracy, feed_dict = {x: images, target: labels, kp: 1.0})
				testAccuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, target: mnist.test.labels, kp: 1.0})
				print("{:5d}: {:7.7f} {:7.7f}".format(e, trainAccuracy, testAccuracy))
	saver.save(sess, "./data/fixedSplineDAE{}-{}-{}/fineTuning.ckpt".format(internalKnots, p, runNumber))
	
	np.save("./data/fixedSplineDAE{}-{}-{}/error".format(internalKnots, p, runNumber), [epochArray, trainAccuracyArray, testAccuracyArray])
	
	# print("Plotting input test images")
	# canvas = np.empty((28*exhibitionSize, 28*exhibitionSize))
	# for i in range(exhibitionSize):
	# 	for j in range(exhibitionSize):
	# 		canvas[i*28:(i + 1)*28, j*28:(j + 1)*28] = displayImages[3*i + j].reshape([28, 28])
	# plt.figure(figsize=(exhibitionSize, exhibitionSize))
	# plt.imshow(1 - canvas, origin="upper", cmap="gray")
	# plt.axis("off")
	# plt.savefig('original.png')
	
	# print("Plotting reconstruction")
	# g = sess.run(decoder2, feed_dict={x: displayImages})
	# for i in range(exhibitionSize):
	# 	for j in range(exhibitionSize):
	# 		# Draw the reconstructed digits
	# 		data = g[3*i + j].reshape([28, 28])
	# 		canvas[i*28:(i + 1)*28, j*28:(j + 1)*28] = data*(data>0)
	# plt.figure(figsize=(exhibitionSize, exhibitionSize))
	# plt.imshow(1 - canvas, origin="upper", cmap="gray")
	# plt.axis("off")
	# plt.savefig('compressedSpline{}.png'.format(hidden))
	