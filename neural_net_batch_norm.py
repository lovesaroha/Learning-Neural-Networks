# Double layer neural network with batch norm.
import numpy
import math

from numpy.core.fromnumeric import shape

# learning rate.
learningRate = 0.5

# Inputs.
inputs = numpy.array([[1, 1, 0, 0], [1, 0, 1, 0]])
# Outputs OR gate.
outputs = [1, 1, 1, 0]

# Layer one (5 units).
w1 = numpy.random.randn(5, 2)
b1 = numpy.random.randn(5, 1)

# Layer two (3 unit).
w2 = numpy.random.randn(3, 5)
beta2 = numpy.random.randn(3, 1)
gama2 = numpy.random.randn(3, 1)
mean2 = 0
var2 = 0

# Layer three (1 unit).
w3 = numpy.random.randn(1, 3)
b3 = numpy.random.randn(1, 1)

# Sigmoid.


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoidV = numpy.vectorize(sigmoid)

# This function compute hypothesis (Forward propogation).


def predict():
    global inputs, w1, b1, z1, a1, w2,  z2, a2, w3,  z3, a3, b3
    # Layer one.
    z1 = numpy.dot(w1, inputs) + b1
    a1 = sigmoidV(z1)

    # Layer two.
    z2 = numpy.dot(w2, a1)
    z2n = numpy.divide(numpy.subtract(z2, mean2), numpy.sqrt(var2))
    z2f = numpy.multiply(gama2, z2n) + beta2
    a2 = sigmoidV(z2f)

    # Layer three.
    z3 = numpy.dot(w3, a2) + b3
    a3 = sigmoidV(z3)


# Train weights and biases using GD (Back Propagation).
for i in range(2000):
    # Predict (Forward Propagation).
    # Layer one.
    z1 = numpy.dot(w1, inputs) + b1
    a1 = sigmoidV(z1)

    # Layer two.
    z2 = numpy.dot(w2, a1)
    mean = numpy.sum(z2, axis=1, keepdims=True) / 4
    var = numpy.sum(numpy.square(numpy.subtract(z2, mean)) ,axis=1, keepdims=True ) / 4
    z2n = numpy.divide(numpy.subtract(z2, mean), numpy.sqrt(var))
    z2f = numpy.multiply(gama2, z2n) + beta2
    a2 = sigmoidV(z2f)
    # Weigthed average of mean and variance.
    mean2 = 0.9 * mean2 + (0.1) * mean
    var2 = 0.9 * var2 + (0.1) * var

    # Layer three.
    z3 = numpy.dot(w3, a2) + b3
    a3 = sigmoidV(z3)

    # Back Propagation.
    # Layer 3.
    dz3 = a3 - outputs
    dw3 = numpy.dot(dz3, a2.T) / 4
    db3 = numpy.sum(dz3, axis=1, keepdims=True) / 4
    w3 -= learningRate * dw3
    b3 -= learningRate * db3
    # Layer 2.
    da2 = numpy.dot(w3.T, dz3)
    dzf2 = numpy.multiply(da2, numpy.multiply(a2, (1-a2)))
    dz2n = numpy.multiply(dzf2, gama2)
    dz2 = numpy.divide(dz2n , var)

    dgama2 = numpy.multiply(dzf2, z2n)
    gama2 -= learningRate * numpy.sum(dgama2, axis=1, keepdims=True) / 4
    beta2 -= learningRate * numpy.sum(dzf2, axis=1, keepdims=True) / 4

    dw2 = numpy.dot(dz2, a1.T) / 4
    w2 -= learningRate * dw2
    # Layer 1.
    da1 = numpy.dot(w2.T, dzf2)
    dz1 = numpy.multiply(da1, numpy.multiply(a1, (1-a1)))
    db1 = numpy.sum(dz1, axis=1, keepdims=True) / 4
    dw1 = numpy.dot(dz1, inputs.T) / 4
    w1 -= learningRate * dw1
    b1 -= learningRate * db1


predict()
print(a3)