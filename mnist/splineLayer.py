import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.framework import ops

def importSpline(path, fixedKnots = False):
		if(sys.platform == "linux" or sys.platform == "linux2"):
			#Ubuntu
			directory = "/ubuntu/"
		elif(sys.platform == "darwin"):
			#MacOS
			directory = "/macos/"
		else:
			print("Unknown OS")
			exit()
		
		if(fixedKnots):
			funcModule = tf.load_op_library(path + directory + "fixedSpline.so")
			gradModule = tf.load_op_library(path + directory + "fixedSplineGrad.so")
			spline = funcModule.fixed_b_spline
			splinegrad = gradModule.fixed_b_spline_grad
			
			@ops.RegisterGradient("FixedBSpline")
			def _bsplinegrad(op, grad):
				gx, gc = splinegrad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3])
				return gx, 0*op.inputs[1], gc, 0*op.inputs[3]
			
			return spline
		else:
			funcModule = tf.load_op_library(path + directory + "spline.so")
			gradModule = tf.load_op_library(path + directory + "splineGrad.so")
			sortModule = tf.load_op_library(path + directory + "sortKnot.so")
			sortgradModule = tf.load_op_library(path + directory + "sortKnotGrad.so")
			spline = funcModule.b_spline
			splinegrad = gradModule.b_spline_grad
			sortKnot = sortModule.sort_knot
			sortgrad = sortgradModule.sort_knot_grad

			@ops.RegisterGradient("BSpline")
			def _bsplinegrad(op, grad):
				gx, gt, gc = splinegrad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3])
				return gx, gt, gc, 0*op.inputs[3]
			
			@ops.RegisterGradient("SortKnot")
			def _sortknotgrad(op, grad):
				g = sortgrad(grad, op.inputs[0], op.inputs[1])
				return g, 0*op.inputs[1];
			
			return spline, sortKnot

def splineParameters(nodes, basisSize, degree, knots = None, name = ""):
	c = tf.Variable(tf.random_normal([nodes, basisSize], stddev = 0.1), name = name + "c")
	t = np.zeros(basisSize + degree + 1)
	#Constructing p+1 regular knot vector with uniformly spaced knots
	t = np.zeros(basisSize + degree + 1)
	t[degree + 1:basisSize] = np.arange(1, basisSize - degree)
	if(type(knots) == list or type(knots) == np.ndarray):
		t[:degree + 1] = knots[0]
		t[degree + 1:basisSize] = knots[0] + (knots[1] - knots[0])*t[degree + 1:basisSize]/(basisSize - degree)
		t[basisSize:] = knots[1]
	else:
		t[degree + 1:basisSize] = np.arange(1, basisSize - degree)
		if(not knots):
			t[basisSize:] = basisSize - degree
		else:
			t[degree + 1:basisSize] = knots*t[degree + 1:basisSize]/(basisSize - degree)
			t[basisSize:] = knots
	t = tf.Variable(np.array(t, dtype = np.float32), name = name + "t")
	return t, c