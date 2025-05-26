# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from neural_network import *

EARLY_STOP_LIMIT = 5
NUMBER_OF_CLASSES = 10

def save(model, filename):
    pickle.dump(model,open(filename, 'wb'))

def load(filename):
    return pickle.load(open(filename, 'rb'))

def early_stopping(loss_list = []):
    if len(loss_list) >= EARLY_STOP_LIMIT + 1:
        recent_loss_val = loss_list[-1]
        count = 0
        for loss in loss_list[-5:]:
            if np.all(recent_loss_val >= loss):
                count+=1
        if count == 5:
            return True
        return False

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def train(sequential_network, x_train, y_train, epochs = 2000, learning_rate = 0.01):
    loss_list =[]
    for itr in range(epochs):
        error = 0
        for x, y_actual in zip(x_train, y_train):
            y_pred = sequential_network.forward(x)
            cross = CrossEntropy()
            error += cross.cross_entropy_loss(y_pred, y_actual)
            gradient_loss = cross.cross_entropy_grad(y_pred, y_actual)
            sequential_network.backward(gradient_loss, learning_rate = 0.01)
        error /= len(x_train)
        loss_list.append(error)
        # Early stopping
        if early_stopping(loss_list):
            epochs = itr + 1
            break
    # fig = plt.figure()
    # ax = fig.add_subplot(211)
    # ax.plot(np.arange(0,epochs), np.array(loss_list).reshape(-1))
    # ax.set_title('Training Loss')
    return sequential_network
    
def plot_validation_loss(model, x_val,y_val):
    epochs = 200
    error_mean = []
    for itr in range(0,epochs):
        error_list = []
        for idx, val_sample in enumerate(x_val):
            val_pred = model.forward(val_sample)
            cross = CrossEntropy()
            error = cross.cross_entropy_loss(val_pred, y_val[idx])
            error_list.append(error)
        error_mean.append(np.mean(error_list))
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(np.arange(0, epochs), np.array(error_mean).reshape(-1))
    ax.set_title('Validation Loss')
    
def predict(model, x_test,y_actual):
    score = 0
    for idx, test_sample in enumerate(x_test):
        y_pred = model.forward(test_sample)
        #print(f'''Test Sample - {test_sample} \n Model Predicted Value- {y_pred}''')
        soft = Softmax()
        probs = soft.forward(y_pred)
        print(f'''Actual value - {y_actual[idx]} Predicted Value - {np.argmax(probs)}''')
        if np.all(np.argmax(probs) == y_actual[idx]):
            score+=1
    return score/len(y_actual)

def preprocess(samples, targets):
    # reshape and normalization
    samples = samples.reshape(samples.shape[0], 28 * 28, 1)
    samples = samples.astype("float32") / 255
    #convery to numbers
    conv_targets = [int(num) for num in targets]
    conv_targets = np.array(conv_targets)
    targets_vect = get_one_hot(conv_targets, NUMBER_OF_CLASSES)
    targets_vect = targets_vect.reshape(targets_vect.shape[0], 10, 1)
    return samples, targets_vect\

data, targets = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

X, Y = preprocess(data, targets)
#Training Split 
train_split, combine_split, train_y_split, combine_y_split = train_test_split(X, Y ,test_size=0.2, random_state=123)
#Validation Split
val_split, test_split, val_y_split, test_y_split = train_test_split(combine_split, combine_y_split, test_size=0.5, random_state=123)

mnist_model_1 = Sequential([
    LinearLayer(28*28, 40), Sigmoid(),
    LinearLayer(40, 10), Sigmoid()])

mnist_model_2 = Sequential([
    LinearLayer(28*28, 40), HyperbolicTangent(),
    LinearLayer(40,10), HyperbolicTangent()])

mnist_model_3 = Sequential([
    LinearLayer(28*28, 40), HyperbolicTangent(),
    LinearLayer(40, 20), HyperbolicTangent(),
    LinearLayer(20, 10), HyperbolicTangent()])

model_1 = train(mnist_model_1, train_split, train_y_split, epochs=10, learning_rate = 0.001)
model_2 = train(mnist_model_2, train_split, train_y_split, epochs=10, learning_rate = 0.001)
model_3 = train(mnist_model_3, train_split, train_y_split, epochs=10, learning_rate = 0.001)

plot_validation_loss(model_1, val_split,val_y_split)
plot_validation_loss(model_2, val_split,val_y_split)
plot_validation_loss(model_3, val_split,val_y_split)

score_model_1 = predict(model_1,test_split,test_y_split)
print('Model_1 Score: ',round(score_model_1,4))

score_model_2 = predict(model_2,test_split,test_y_split)
print('Model_2 Score: ',round(score_model_2,4))

score_model_3 = predict(model_3,test_split,test_y_split)
print('Model_3 Score: ',round(score_model_3,4))

mnist_file1 = 'MNIST_model1.w'
save(model_1, mnist_file1)

mnist_file2 = 'MNIST_model2.w'
save(model_2, mnist_file2)

load_model1 = load(mnist_file1)
load_model2 = load(mnist_file2)