# BsplineNetworks

The code for performing the experiments discussed in my masters thesis in applied mathematics.

The code is written in Python using Tensorflow, and implements spline activation functions in neural networks. The spline activation functions are implemented as B-splines arbitrary polynomial degree with fixed and free knots.

Spline activation functions for neural networks is located in [splineOp](./splineOp). The code for testing convergence to a spline function is also in this directory. ([splineFitting.py](./splineOp/splineFitting.py)) Two bash scripts are included to compile against macOS and Ubuntu.

A spline neural network which performs regression on the California housing dataset is located in [housing](./housing).

Spline networks which investigates the MNIST dataset is located in [mnist](./mnist).
