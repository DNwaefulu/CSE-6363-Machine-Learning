# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:28:02 2023

@author: dnwae
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.train_losses = []
        self.value_losses = []

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------1
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # TODO: Initialize the weights and bias based on the shape of X and y.
        self.weights = np.random.uniform(-1, 1, size=(X.shape[1]))
        self.bias = np.random.uniform(-1, 1, size=(1))

        # TODO: Implement the training loop.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=45, shuffle=True, stratify=None)
        
        current = 0
        mse_loss = float('inf')
        previous_weights = []
        mse_results = []
        
        
        """
        100 max epochs
        """
        for n in range(self.max_epochs):
            if current >= patience:
                self.weights = previous_weights[0]
                break
            
            for i in range(0, X_train.shape[0], batch_size):
                batch_data_X = X_train[i:i+batch_size]
                batch_data_y = y_train[i:i+batch_size]
                
                prediction = batch_data_X.values @ self.weights
                error = prediction - batch_data_y.values
                
                """
                Gradient Descent
                """
                gradient = error @ batch_data_X
                self.weights = self.weights - (0.001 * gradient)
                
            train_prediction = X_train @ self.weights
            train_error = train_prediction - y_train
            train_mse = (train_error**2).mean()
            self.train_losses.append(train_mse)
            
            value_prediction = X_test @ self.weights
            value_error = value_prediction - y_test
            value_mse = (value_error**2).mean()
            self.value_losses.append(value_mse)
            
            if mse_loss > value_mse:
                current = 0
            else:
                previous_weights.append(self.weights)
                current = current + 1
            mse_loss = value_mse
            
            mse_results.append(mse_loss)
            steps = n
            
            
            print("Validation Mean Squared Error: {} for step {}".format(mse_loss, n+1))
        print("Final Weights:  ",self.weights)
        return mse_results, steps

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        prediction = X @ self.weights
        return prediction

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        prediction = X @ self.weights
        error = prediction - y.values
        mse_loss = (error**2).mean()
        return mse_loss
    
    def saveLoad(self, filename):
        f_name = filename
        pickle.dump(self.train_losses, open(f_name, 'wb'))
        pickle.dump(self.value_losses, open(f_name, 'wb'))
        load_model = pickle.load(open(filename, 'rb'))
        return load_model