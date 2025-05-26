# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 01:23:04 2023

@author: dnwae
"""

import numpy as np

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Initialize weights and bias
        self.weights = np.random.randn(num_features)
        self.bias = np.random.randn()

        # Early stopping setup
        best_loss = float('inf')
        consecutive_increases = 0

        for epoch in range(self.max_epochs):
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                # Calculate predictions and errors
                y_pred = X_batch.dot(self.weights) + self.bias
                errors = y_pred - y_batch

                # Calculate gradients
                gradient_weights = 2 * X_batch.T.dot(errors) / self.batch_size
                gradient_bias = 2 * np.sum(errors) / self.batch_size

                # Update weights and bias with regularization
                self.weights -= self.regularization * self.weights - gradient_weights
                self.bias -= gradient_bias

            # Calculate validation loss
            val_predictions = self.predict(X)
            val_loss = np.mean((val_predictions - y) ** 2)

            print(f"Epoch {epoch+1}/{self.max_epochs} - Validation loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                consecutive_increases = 0
            else:
                consecutive_increases += 1
                if consecutive_increases >= self.patience:
                    print("Early stopping triggered.")
                    break

    def predict(self, X):
        return X.dot(self.weights) + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        return mse