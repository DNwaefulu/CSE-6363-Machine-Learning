# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:24:55 2023

@author: dnwae
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from sklearn.datasets import fetch_openml


"""
For the layers you will create in this assignment, it is worth it to create a 
parent class named Layer which defined the forward and backward functions that are used by all layers.
"""

class Layer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x):
        pass
    
    def backward(self, gradient, learning_rate):
        pass
    
"""
Create a class that implements a linear layer. The class should inherit the Layer class and implement both a 
forward and backward function.
"""

class LinearLayer(Layer):
    def __init__(self, size_x, size_y):
        self.weights = np.random.rand(size_y, size_x)
        self.bias = np.random.rand(size_y, 1)
    
    # f(x; w) = xw^T + b
    def forward(self, x):
        self.x = x
        return np.dot(self.weights, self.x) + self.bias
    
    def backward(self, gradient, learning_rate = 0.01):
        weights_gradient = np.dot(gradient, self.x.T)
        gradient_x = np.dot(self.weights.T, gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * gradient
        return gradient_x

"""
Create a class that implements the logistic sigmoid function. The class should inherit the Layer class 
and implement both forward and backward functions.
"""    
class Sigmoid(Layer):
    def __init__(self):
        self.sigmoid = lambda x : 1 / (1 + np.exp(-x))
        self.sigmoid_gradient = lambda x : self.sigmoid(x) * (1 - self.sigmoid(x))
    
    # f(x) = 1 / (1 + e^-x) 
    def forward(self, x):
        self.x = x
        return self.sigmoid(self.x)
    
    # df/dx = f(x)(1 - f(x))
    def backward(self, gradient, learning_rate = 0.01):
        return np.multiply(gradient, self.sigmoid_gradient(self.x))

"""
Create a class that implements the hyperbolic tangent function. The class should inherit the Layer class 
and implement both forward and backward functions.
"""        
class HyperbolicTangent(Layer):
    def __init__(self):
        self.tanh = lambda x : np.tanh(x)
        self.tahn_gradient = lambda x : 1 - np.tanh(x)**2
        
    # f(x) = e^x - e^-x / e^x - e^-x  (tangent)
    def forward(self, x):
        self.x = x
        self.result = self.tanh(self.x)
        return self.result
    
    # df/dx = 1 - f(x)^2
    def backward(self, gradient, learning_rate = 0.01):
        self.result = np.multiply(gradient,self.tahn_gradient(self.x))
        return self.result

"""
Create a class that implements the softmax function. The class should inherit the Layer class 
and implement both forward and backward functions.
"""    
class Softmax(Layer):   
    def forward(self, x):
        self.x = x
        num = np.exp(self.x)
        return num / np.sum(num, axis=0, keepdims=True)
    
    def backward(self, x):
        return x

"""
Create a class that implements cross-entropy loss. The class should inherit the Layer class 
and implement both forward and backward functions.
"""      
class CrossEntropy(Layer):
    def cross_entropy_loss(self, pred, target):
        return -target * np.log(pred)
    
    def cross_entropy_grad(self, pred, target):
        return target - pred
    
    def forward(self, pred, target):
        self.p = pred
        self.t = target
        return self.cross_entropy_loss(self.t, self.p)
    
    def backward(self, pred, target):
        self.p = pred
        self.t = target
        return self.cross_entropy_grad(self.t, self.p)

"""
In order to create a clean interface that includes multiple layers, you will need to create a class
that contains a list of layers which make up the network. The Sequential class will contain a list of layers.
New layers can be added to it by appending them to the current list.
This class will also inherit from the Layer class so that it can call forward and backward as required.
"""
    
class Sequential(Layer):
    def __init__(self, network = []):
        self.network = network
        
    def forward(self, x):
        result = x
        for layer in self.network:
            result = layer.forward(result)
        return result
    
    def backward(self, gradient, learning_rate = 0.01):
        grad = gradient
        for layer in reversed(self.network):
            grad = layer.backward(grad, learning_rate)
        return grad