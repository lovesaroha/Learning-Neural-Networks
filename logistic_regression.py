# Logistic regression.

import numpy
import math

# learning rate.
learningRate = 0.2

# Weights and biases.
w = numpy.random.randn(2, 1)
b = numpy.random.randn(1, 1)

# Inputs.
inputs = numpy.array([[1, 1, 0, 0], [1, 0, 1, 0]])

# Outputs OR gate.
outputs = [1, 1, 1, 0]

# This function compute hypothesis.


def predict(input):
    print(sigmoidV(numpy.dot(w.T, input) + b))

# Sigmoid.


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoidV = numpy.vectorize(sigmoid)

# Train weights and biases using GD.
for i in range(1000):
    z = numpy.dot(w.T, inputs) + b
    a = sigmoidV(z)
    dz = a - outputs
    b -= learningRate * numpy.sum(dz) / 4
    w -= learningRate * (numpy.dot(inputs, dz.T) / 4)


predict(inputs)
