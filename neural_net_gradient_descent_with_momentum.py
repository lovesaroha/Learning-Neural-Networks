# Single layer neural network using gradient descent with momentum.
import numpy
import math

from numpy.core.fromnumeric import shape

# learning rate.
learningRate = 0.2

# Inputs.
inputs = numpy.array([[1, 1, 0, 0], [1, 0, 1, 0]])
# Outputs OR gate.
outputs = [1, 1, 1, 0]

# Layer one (2 units).
w1 = numpy.random.randn(2, 2)
b1 = numpy.random.randn(2, 1)

# Layer two (1 unit).
w2 = numpy.random.randn(1, 2)
b2 = numpy.random.randn(1, 1)

# For momentum.
beta = 0.9
vdw1 = numpy.zeros((2,2))
vdb1 = numpy.zeros((2,1))
vdw2 = numpy.zeros((1,2))
vdb2 = numpy.zeros((1,1))



# Sigmoid.


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoidV = numpy.vectorize(sigmoid)

# This function compute hypothesis (Forward propogation).


def predict():
    global inputs, w1, b1, z1, a1, w2, b2, z2, a2
    # Layer one.
    z1 = numpy.dot(w1, inputs) + b1
    a1 = sigmoidV(z1)
    # Layer two.
    z2 = numpy.dot(w2, a1) + b2
    a2 = sigmoidV(z2)


# Train weights and biases using GD (Back Propagation).
for i in range(1000):
    predict()
    # Layer 2.
    dz2 = a2 - outputs
    dw2 = numpy.dot(dz2, a1.T) / 4
    db2 = numpy.sum(dz2, axis=1, keepdims=True) / 4
    # Weighted average.
    vdw2 = numpy.multiply(beta, vdw2) + numpy.multiply(1-beta, dw2)
    vdb2 = numpy.multiply(beta, vdb2) + numpy.multiply(1-beta, db2)
    w2 -= learningRate * vdw2
    b2 -= learningRate * vdb2

    # Layer 1.
    da1 = numpy.dot(w2.T, dz2)
    dz1 = numpy.multiply(da1, numpy.multiply(a1, (1-a1)))
    dw1 = numpy.dot(dz1, inputs.T) / 4
    db1 = numpy.sum(dz1, axis=1, keepdims=True) / 4
    # Weighted average.
    vdw1 = numpy.multiply(beta, vdw1) + numpy.multiply(1-beta, dw1)
    vdb1 = numpy.multiply(beta, vdb1) + numpy.multiply(1-beta, db1)
    w1 -= learningRate * vdw1
    b1 -= learningRate * vdb1

predict()
print(a2)
