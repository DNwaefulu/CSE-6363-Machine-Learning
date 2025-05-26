# -*- coding: utf-8 -*-

import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
from neural_network import *

def save(model, filename):
    pickle.dump(model,open(filename, 'wb'))

def load(filename):
    return pickle.load(open(filename, 'rb'))

def train(sequential_network, x_train, y_train, epochs = 2000, learning_rate = 0.01):
    loss_list =[]
    for itr in range(epochs):
        error = 0
        for x, y_actual in zip(x_train, y_train):
            y_pred = sequential_network.forward(x)
            #print('y---pred',y_pred)
            cross = CrossEntropy()
            error += cross.cross_entropy_loss(y_actual, y_pred)
            #print('error: ',error)
            gradient_loss = cross.cross_entropy_grad(y_actual, y_pred)
            sequential_network.backward(gradient_loss, learning_rate = 0.01)
        error /= len(x_train)
        loss_list.append(error)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    #ax.plot(np.arange(0,epochs),np.array(loss_list).reshape(-1))
    ax.set_title('Training Loss')
    return sequential_network


def predict_xor(model, x_test,y_actual):
    score = 0
    for idx, test_sample in enumerate(x_test):
        y_pred = model.forward(test_sample)
        #print(f'''Test Sample - {test_sample} \n Model Predicted Value- {y_pred}''')
        soft = Softmax()
        probs = soft.forward(y_pred)
        print(f'''Actual value - {y_actual[idx]} Predicted Value - {np.argmax(probs)}''')
        if np.argmax(probs) == y_actual[idx]:
            score+=1
    return score/len(y_actual)

samples = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
targets = np.array([0, 1, 1, 0])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(samples[:, 0], samples[:, 1], c=targets)
ax.set_title('XOR')

model1 = Sequential([
    LinearLayer(2, 2), Sigmoid(),
    LinearLayer(2, 1), Sigmoid()])

model2 = Sequential([
    LinearLayer(2, 2), HyperbolicTangent(),
    LinearLayer(2, 1), HyperbolicTangent()])

trained_model = train(model1, samples.reshape(4,2,1), targets.reshape(4,1,1), epochs=2000)

filename = 'XOR_solved.w'
save(trained_model, filename)
mp = load(filename)

print(mp)

score = predict_xor(trained_model, samples.reshape(4, 2, 1), targets.reshape(4, 1, 1))
print('Model Score: ', score)

print("\nLoaded weights\n")
predict_xor(mp,samples.reshape(4,2,1),targets)
targets.reshape(4,1,1)[3]
