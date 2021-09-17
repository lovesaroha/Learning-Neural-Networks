# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Single layer neural network.
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
    w2 -= learningRate * dw2
    b2 -= learningRate * db2
    # Layer 1.
    da1 = numpy.dot(w2.T, dz2)
    dz1 = numpy.multiply(da1, numpy.multiply(a1, (1-a1)))
    dw1 = numpy.dot(dz1, inputs.T) / 4
    db1 = numpy.sum(dz1, axis=1, keepdims=True) / 4
    w1 -= learningRate * dw1
    b1 -= learningRate * db1

predict()
print(a2)
