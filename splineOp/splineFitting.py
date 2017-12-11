import sys
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import numpy as np
import time
from splineLayer import importSpline, splineParameters
import time
import seaborn as sns
sns.set_style("ticks")
sns.set_context("poster")
spline, sortKnot = importSpline(".")

epochs = 100000
samples = 1000
dstep = 1000
p = 3
n = 15
nodes = 1

def compSpline(x, t, c, p):
	y = np.empty([len(c), len(x)])
	for i in range(len(c)):
		y[i] = np.empty(len(x))
		y[i][x < t[0]] = c[i][0]
		y[i][x >= t[-p]] = c[i][-1]
		tmp = t[0] <= x
		tmp = tmp*(x < t[-p])
		y[i][tmp] = BSpline(t, c[i], p)(x[tmp])
	return y.T

coeffs = 4*np.random.random(n) - 2
splineSpace = len(coeffs)
coeffs = np.array([coeffs])
knots = np.zeros(n + p + 1)
knots[:p+1] = 0
knots[-p-1:] = n - p
knots[p+1:-p-1] = np.sort(np.random.random(len(knots[p+1:-p-1]))*(knots[-1] - knots[0])) + knots[0]
sampleStep = int((epochs - 1)/(samples - 1))

x = tf.placeholder(tf.float32, [None, 1])
z = tf.placeholder(tf.float32, [None, 1])

tz, c = splineParameters(nodes, splineSpace, p, [knots[0], knots[-1]])

t = sortKnot(tz, p)
y = spline(x, t, c, p)

loss = tf.reduce_mean((y - z)**2)
optimizer = tf.train.AdamOptimizer(2e-3).minimize(loss)

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 12
config.inter_op_parallelism_threads = 12

tees = np.empty([samples, len(knots)])
cees = np.empty([samples, len(coeffs.T)])
tzees = np.empty([samples, len(knots)])
errors = np.empty(samples)
epochArray = np.empty(samples)

#Create train and test data
samples = 2500
split = 0.3
indexes = np.arange(samples)
np.random.shuffle(indexes)
rawInputs = np.linspace(knots[0], knots[-1], samples)
rawTargets = compSpline(rawInputs, knots, coeffs, p)
trainInputs = rawInputs[indexes[int(samples*split):]][:, np.newaxis]
testInputs =  rawInputs[indexes[:int(samples*split)]][:, np.newaxis]
trainTargets = rawTargets[indexes[int(samples*split):]]
testTargets =  rawTargets[indexes[:int(samples*split)]]

with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	t0 = time.time()
	tt0 = t0
	for e in range(epochs):
		sess.run(optimizer, feed_dict = {x: trainInputs, z: trainTargets})
		if(e%sampleStep == 0):
			i = e//sampleStep
			tees[i] = sess.run(t)
			tzees[i] = sess.run(tz)
			cees[i] = sess.run(c)
			errors[i] = sess.run(loss, feed_dict = {x: trainInputs, z: trainTargets})
			epochArray[i] = e
		if(e%dstep == 0):
			t1 = time.time()
			l = sess.run(loss, feed_dict = {x: trainInputs, z: trainTargets})
			print(e, l, t1 - tt0)
			tt0 = t1
	print(time.time() - t0)
	
	inp = np.linspace(knots[0], knots[-1], samples + 1)[:-1]
	target = compSpline(inp, knots, coeffs, p)
	y = sess.run(y, feed_dict = {x: inp[:, np.newaxis]})
	
	plt.figure(1)
	plt.plot(epochArray, tees, "-b")
	plt.plot([epochs], knots[:, np.newaxis].T, "or")
	plt.xlabel("Training epochs")
	plt.title("Knots")
	sns.despine()
	plt.gcf().subplots_adjust(bottom = 0.2)
	plt.gcf().subplots_adjust(left = 0.2)
	plt.figure(2)
	plt.plot(epochArray, cees, "-b")
	plt.plot([epochs], coeffs, "or")
	plt.xlabel("Training epochs")
	plt.title("Control points")
	sns.despine()
	plt.gcf().subplots_adjust(bottom = 0.2)
	plt.gcf().subplots_adjust(left = 0.2)
	plt.figure(3)
	plt.semilogy(epochArray, errors, "-b")
	plt.xlabel("Training epochs")
	plt.title("Mean squared error", y = 0.9)
	sns.despine()
	plt.gcf().subplots_adjust(bottom = 0.2)
	plt.gcf().subplots_adjust(left = 0.2)
	plt.figure(4)
	plt.plot(inp, target - y, '-b')
	plt.title("Spline curve error", y = 0.95)
	sns.despine()
	plt.gcf().subplots_adjust(bottom = 0.2)
	plt.gcf().subplots_adjust(left = 0.2)
	
	
	tstar = np.array([np.sum(knots[j + 1: j + p + 1]) for j in range(n)])/p
	tstarp = np.array([np.sum(tees[-1][j + 1: j + p + 1]) for j in range(n)])/p
	plt.figure(5)
	plt.plot(tstar, coeffs[0], "--r")
	plt.plot(inp, target, "-r")
	plt.plot(tstarp, cees[-1], "--b")
	plt.plot(inp, y, "-b")
	plt.title("Spline curve with control polygon")
	sns.despine()
	plt.gcf().subplots_adjust(bottom = 0.2)
	plt.gcf().subplots_adjust(left = 0.2)
	plt.figure(6)
	plt.plot(inp, target, "-r")
	plt.plot(inp, y, "-b")
	plt.legend(["Target", "Model"])
	plt.title("Spline curve")
	
	sns.despine()
	plt.gcf().subplots_adjust(bottom = 0.2)
	plt.gcf().subplots_adjust(left = 0.2)
	plt.figure(7)
	plt.plot(tstar, coeffs[0], "-sr")
	plt.plot(tstarp, cees[-1], "-ob")
	plt.title("Control polygon")
	plt.legend(["Target", "Model"])
	sns.despine()
	plt.gcf().subplots_adjust(bottom = 0.2)
	plt.gcf().subplots_adjust(left = 0.2)
	
	
	
	
	plt.figure(15)
	plt.subplot(221)
	plt.plot(inp, target, '-b')
	plt.plot(inp, y, '-r')
	plt.title("Spline curve")
	plt.subplot(222)
	plt.plot(epochArray, tees)
	plt.title("Knots")
	plt.subplot(223)
	plt.plot(epochArray, cees)
	plt.title("Coefficients")
	plt.subplot(224)
	plt.plot(epochArray, tzees)
	plt.title("Zero knots")
	plt.show()