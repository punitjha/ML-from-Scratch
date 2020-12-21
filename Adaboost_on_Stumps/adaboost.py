# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:00:23 2020

@author: Punit
"""


import numpy as np
import matplotlib.pyplot as plt

xs = np.array([
    [0., -.9],
    [0.0, -0.1],
    [-.2, -.4],
    [-.9, -.7],
    [-.9, .5],
    [.98, 0.],
    [.8, .5],
    [.7, .7],
    [.4, -.2],
    [.3, -.7]
    ])
ys = np.array([1, -1, 1, -1, -1, 1, 1, -1, -1, 1])

def plot_stumps(data, labels, stumps, classifier_weights=None):
    agg_pred = np.zeros(labels.shape)
    if classifier_weights is None:
        classifier_weights = [1] * labels.size
    for s, alpha in zip(stumps, classifier_weights):
        agg_pred += alpha * s.predict(data)
    #ties assumed positive
    agg_pred = np.sign(np.sign(agg_pred) + 0.01)
    fig, axes = plt.subplots(2)
    axes[0].set_title("Actual")
    axes[0].plot(data[labels == 1, 0], data[labels == 1, 1], 'rx')
    axes[0].plot(data[labels == -1, 0], data[labels == -1, 1], 'bo')
    axes[1].set_title("Predicted")
    axes[1].plot(data[agg_pred == 1, 0], data[agg_pred == 1, 1], 'rx')
    axes[1].plot(data[agg_pred == -1, 0], data[agg_pred == -1, 1], 'bo')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
    
    
import numpy as np
import matplotlib.pyplot as plt
import copy

class Stump:
    def __init__(self, data, labels, weights=None):
        """
        Initialize a stump that minimizes a weighted error function
        Assume we are working over data [-1, 1]^2
        :param data: numpy array of shape (N, 2)
        :param labels: numpy array of shape (N,) consisting of values in {-1, 1}
        :param weights: numpy array of shape (N,)
        :returns None
        """
       
        self.weights=weights
        self.best_dimension = 0
        self.best_direction = 1
        self.best_pos = 0
        self.min_error = np.sum(weights)
        
        for x in range(data.shape[1]):
            data_slice=data[:,x]
            y=labels
            min_x, max_x = data_slice.min(), data_slice.max()
            len_x = max_x - min_x
            slice_min_error = np.sum(weights)
            steps=10
            slice_best_direction=0
            slice_best_pos=data_slice[0]
            for position in np.arange(min_x, max_x, len_x/steps):
                for directions in [-1, 1]:
                    gy = np.ones((y.size))
                    gy[data_slice*directions < position*directions] = -1
                    slice_error = np.sum((gy != y)*weights)
                    if slice_error < slice_min_error:
                        slice_min_error = slice_error
                        slice_best_direction = directions
                        slice_best_pos = position
            if slice_min_error < self.min_error:
                self.min_error = slice_min_error
                self.best_direction = slice_best_direction
                self.best_pos = slice_best_pos
                self.best_dimension=x


    def predict(self, data):
        """
        Initialize a stump that minimizes a weighted error function
        Assume we are working over data [-1, 1]^2
        :param data: numpy array of shape (N, 2)
        :returns numpy array of shape (N,) containing predictions (in {1,-1})
        """
        predictions = np.ones(data.shape[0])
        data_slice=data[:,self.best_dimension]
        predictions[data_slice*self.best_direction < self.best_pos*self.best_direction] = -1
        return predictions


def adaboost(data, labels, n_classifiers):
    """
    Run the adaboost algorithm
    :param data: numpy array of shape (N, 2)
    :param labels: numpy array of shape (N,), containing values in {1,-1}
    :param n_classifiers: number of weak classifiers to learn
    :returns a tuple (classifiers, weights) consisting of a list of stumps and a numpy array of weights for the classifiers
    """
    weights=np.asarray([1/data.shape[0]]*data.shape[0])
    
    classifier_weights = []
    classifiers=[]
    for objs in range(n_classifiers):
        classifiers.append(Stump(xs,ys,weights))
        prediction=classifiers[objs].predict(data)
        z=np.sum((prediction != labels)*weights)
        beta=0.5*np.log((1+z)/(1-z))
        weights=weights*np.exp(-beta*labels*prediction)
        weights=weights/sum(weights)
        classifier_weights.append(beta)    
    return classifiers, classifier_weights
print("No .of classifiers is 1")
classifiers, classifier_weights = adaboost(xs, ys,1)
plot_stumps(xs, ys, classifiers, classifier_weights)
print("No .of classifiers is 2")
classifiers, classifier_weights = adaboost(xs, ys,2)
plot_stumps(xs, ys, classifiers, classifier_weights)
print("No .of classifiers is 5")
classifiers, classifier_weights = adaboost(xs, ys,5)
plot_stumps(xs, ys, classifiers, classifier_weights)