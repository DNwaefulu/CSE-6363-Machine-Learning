# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:51:50 2023

@author: dnwae
"""
import numpy as np
import pandas as pd
import csv

class DecisionTreeNode:
    def __init__(self, feature_index=None, split_value=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.split_value = split_value      # Value of the feature to split on
        self.left = left                    # Left child node
        self.right = right                  # Right child node
        self.value = value                  # Value if the node is a leaf (class label for classification)

class DecisionTree:
    def __init__(self, criterion, max_depth, min_samples_split, min_samples_leaf):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    
    def build_tree(self, dataset, sample_weights, depth):
        
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = np.shape(X)
            
        # Flatten the y array to ensure it is 1-dimensional and of integer type
        y = np.array(y, dtype='int64').flatten()
        
            
        if depth == self.max_depth or len(y) < self.min_samples_split or np.unique(y).size == 1:
            class_counts = np.bincount(y, sample_weights)
            value = np.argmax(class_counts)
            return DecisionTreeNode(value = value)
            
        best_feature, best_threshold = self._split(X, y, sample_weights)
        
        if best_feature is None:
            _class_counts = np.bincount(y, sample_weights)
            _value = np.argmax(_class_counts)
            return DecisionTreeNode(value = _value)
            
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        
        left_child = self.build_tree(dataset[left_idxs], sample_weights[left_idxs], depth =+ 1)
        right_child = self.build_tree(dataset[right_idxs], sample_weights[right_idxs], depth =+ 1)
        
        return DecisionTreeNode(feature_index=best_feature, split_value=best_threshold, left=left_child, right=right_child)
    
    def fit(self, X, y, sample_weights = None):
        # Train tree
        if sample_weights is None:
            sample_weights = np.ones(np.shape(X[0])) / np.shape(X[0])
        
        dataset = np.column_stack((X, y))
        self.root = self.build_tree(dataset, sample_weights, depth = 0)
        
    def predict(self, x):
        return np.array([self._predict(samples) for samples in x])
    
    def _predict(self, inputs):
        node = self.tree_
        while isinstance(node, dict):
            if inputs[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node
        
    def cprob(data, k):
        return np.sum(data == k) / np.shape(data)

    def misclassification_rate(preds, target):
        pi_target = DecisionTree.cprob(preds, target)
        return 1 - pi_target

    def entropy(targets, n_classes):
        """Entropy measures disorder or uncertainty.
        """
        result = 0
        for i in range(len(n_classes)):
            pi_c = DecisionTree.cprob(n_classes, i)
            if pi_c > 0:
                result -= pi_c * np.log2(pi_c)
                return result

    def gini_index(targets, n_classes):
        sum = 0
        for i in range(len(n_classes)):
            pi_c = DecisionTree.cprob(n_classes, i)
            sum += pi_c**2

        return 1 - sum
    
    def _split(self, X, y, sample_weights):

        best_feature, best_threshold = None, None
        best_criterion = float('inf')
        
        for idx in range(X.shape[1]):
            thresholds = np.unique(X[:, idx])
            for threshold in thresholds:
                left_idxs = X[:, idx] <= threshold
                right_idxs = ~left_idxs
                if len(np.array(y)[left_idxs]) < self.min_samples_leaf or len(np.array(y)[right_idxs]) < self.min_samples_leaf:
                    continue
                criterion = self.compute_criterion(y[left_idxs], y[right_idxs], sample_weights, sample_weights)
                    
                if criterion < best_criterion:
                    best_feature = idx
                    best_threshold = threshold
                    best_criterion = criterion
                        
        return best_feature, best_threshold


    def compute_criterion(self, y_left, y_right, weights_left, weights_right):
        n_left = np.size(y_left)
        n_right = np.size(y_right)
        n_total = n_left + n_right
        
        
        if self.criterion == 'gini':
            left_criterion = self.gini_index(y_left)
            right_criterion = self.gini_index(y_right)
            criterion = (n_left / n_total) * left_criterion + (n_right / n_total) * right_criterion
        elif self.criterion == 'entropy':
            left_criterion = self.entropy(y_left)
            right_criterion = self.entropy(y_right)
            criterion = (n_left / n_total) * left_criterion + (n_right / n_total) * right_criterion
        else:
            left_criterion = self.misclassification_rate(y_left, weights_left)
            right_criterion = self.misclassification_rate(y_right, weights_right)
            criterion = (n_left / n_total) * left_criterion + (n_right / n_total) * right_criterion
        
        return criterion


    
class RandomForest:
    def __init__(self, classifier, num_trees, min_features):
        self.classifier = classifier
        self.num_trees = num_trees
        self.min_features = min_features
        self.trees = []
        
    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        for i in range(self.num_trees):
            # Randomly select a subset of data with replacement
            indx = np.random.choice(n_features, size = self.min_features, replace = False)
            X_subset = np.array(X)[indx]
            y_subset = np.array(y)[indx]
            
            # Randomly select a subset of features
            feature_index = np.random.choice(X.shape[1], self.min_features, replace = False)
            X_subset = X_subset[:, feature_index]
            
            # Train a decision tree on the subset of data
            tree = self.classifier()
            tree.fit(X_subset, y_subset)
            
            # Add the trained tree to the list of trees in the forest
            self.trees.append(tree)
        
    def predict(self, X):
        N = np.zeros((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            # Make predictions with each tree in the forest
            N = [tree.predict(X[:, self.indx]) for tree, features in self.trees]
        y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr = N)
        return y_pred
            
        
class AdaBoost:
    def __init__(self, weak_learner, num_learners, learning_rate):
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.learners = []
        self.weights = []
        
    def predict(self, x):
        prediction = np.zeros(x.shape[0])
        for learner, alpha in zip(self.learners, self.weights):
            prediction += alpha * learner.predict(x)
        return np.sign(prediction)
    
    def fit(self, X, y):
        n = X.shape[0]
        self.weights = np.ones(n) / n
        
        for num in range(self.num_learners):
            learner = self.weak_learner
            learner.fit(X, y, self.weights)
            predictions = learner.predict(X)
            error = np.sum(self.weights * (predictions != y))
            
            if error == 0:
                self.learners.append(learner)
                self.weights.append(1.0)
                break
            
            alpha = self.learning_rate * np.log((1 - error) / error)
            
            weight = self.learning_rate * np.exp(-alpha * y * predictions)
            self.weights /= np.sum(self.weights)
            
            self.learners.append(learner)
            self.weights.append(weight)
            
            
def load_data_train(filename):
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            # Preprocess data
            pclass = int(row[2])
            sex = 1 if row[4] == 'female' else -1
            age = float(row[5]) if row[5] != '' else None
            sibsp = int(row[6])
            parch = int(row[7])
            fare = float(row[9]) if row[9] != '' else None
            embarked = ord(row[11]) if row[11] != '' else None
            
            X.append([pclass, sex, age, sibsp, parch, fare, embarked])
            y.append(int(row[1]))
            
    #Input missing values
    X = np.array(X)
    age_mean = np.nanmean(X[:, 2].astype('float64'))
    fare_mean = np.nanmean(X[:, 5].astype('float64'))
    X[:, 2][pd.isna(X[:, 2])] = age_mean
    X[:, 5][pd.isna(X[:, 5])] = fare_mean
    X[:, 6][pd.isna(X[:, 6])] = np.nanmax(X[:, 6].astype('float64')) + 1
    
    return X, y

def load_data_test(filename):
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            # Preprocess data
            pclass = int(row[1])
            sex = 1 if row[3] == 'female' else -1
            age = float(row[4]) if row[4] != '' else None
            sibsp = int(row[5])
            parch = int(row[6])
            fare = float(row[8]) if row[8] != '' else None
            embarked = ord(row[10]) if row[10] != '' else None
            
            X.append([pclass, sex, age, sibsp, parch, fare, embarked])
            y.append(int(row[1]))
            
    #Input missing values
    X = np.array(X)
    age_mean = np.nanmean(X[:, 2].astype('float64'))
    fare_mean = np.nanmean(X[:, 5].astype('float64'))
    X[:, 2][pd.isna(X[:, 2])] = age_mean
    X[:, 5][pd.isna(X[:, 5])] = fare_mean
    X[:, 6][pd.isna(X[:, 6])] = np.nanmax(X[:, 6].astype('float64')) + 1
    
    return X, y


X_train, y_train = load_data_train('train.csv')
X_test, y_test = load_data_test('test.csv')

# Train decision tree classifier
dt = DecisionTree(criterion = 'gini', max_depth = 3, min_samples_split = 2, min_samples_leaf = 5)
dt.fit(X_train, y_train)
dt_acc = np.mean(dt.predict(X_test) == y_test)

# # Train random forest classifier
rf = RandomForest(classifier = DecisionTree(criterion = 'gini', max_depth = 3, min_samples_split = 2, min_samples_leaf = 5), num_trees = 10, min_features = 6)
rf.fit(X_train, y_train)
rf_acc = np.mean(rf.predict(X_test) == y_test)

# Train AdaBoost classifier
ada = AdaBoost(weak_learner = DecisionTree(criterion = 'gini', max_depth = 3, min_samples_split = 2, min_samples_leaf = 5), num_learners = 10, learning_rate = 0.1)
ada.fit(X_train, y_train)
ada_acc = np.mean(ada.predict(X_test) == y_test)



print('Decision tree accuracy:', dt_acc)
print('Random forest accuracy:', rf_acc)
print('AdaBoost accuracy:', ada_acc)

