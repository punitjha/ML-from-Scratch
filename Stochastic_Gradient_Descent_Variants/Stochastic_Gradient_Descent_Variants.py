# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 00:04:20 2020

@author: Punit
"""


## Setup code
import numpy as np
from io import BytesIO

bio = BytesIO(data_files["homeworks/hw1/data/sgd_data.npy"])
data_points = np.load(bio).astype(np.float)
SEED = 1892 # sanity check seed; a different seed will be used for testing
TOL  = 1e-4 # tolerance for testing if things are close

# these weights are meant to sanity check your work
sgd_weights = np.asarray([-3.07246152, 1.65842967, -0.20297172])
sgd_nesterov_weights = np.asarray([-4.36244267, 2.24894154, -0.59244263])
adam_weights = np.asarray([-4.04210889, 2.31674972, -0.25094384])
init_loss = 0.8681132164540256
init_grad = np.asarray([0.78737113, -0.16426571, -0.51562791])
sanity_check = {'sgd' : sgd_weights, 'nesterov' : sgd_nesterov_weights, 'adam' : adam_weights, 'loss' : init_loss, 'grad' : init_grad}


import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    """
    Sigmoid function
    @param  z: input (NumPy array)
    @return sigmoid: sigmoid output (float)
    """
    return 1.0 / (1.0 + np.exp(-z))

def prob(y, x, w):
    """
    Probability estimation
    @param  y: label (integer)
    @param  x: input (NumPy array)
    @param  w: weights (NumPy array)
    @return prob: probability estimation (float)
    """
    return sigmoid(y * np.dot(x, w))

def prob_derivative(y, x, w):
    """
    Gradient of probability estimation
    @param  y: label (integer)
    @param  x: input (NumPy array)
    @param  w: weights (NumPy array)
    @return grad: gradient of probability estimation (NumPy array)
    """
    p = prob(y, x, w)
    return p * (1.0 - p) * y * x



def gradient_loss(y, x, w):
    """
    Gradient of loss function
    @param  y: label (integer)
    @param  x: input (NumPy array)
    @param  w: weights (NumPy array)
    @return grad: gradient of loss (NumPy array)
    """
    # YOUR CODE GOES HERE
    dw=(-y*x*np.exp(-y*(x.dot(w))))/(1+np.exp(-y*x.dot(w)))  
    return dw

def loss_fn(y, x, w):
    """
    Loss function
    @param  y: label (NumPy array)
    @param  x: input (NumPy array)
    @param  w: weights (NumPy array)
    @return loss: negative log likelihood (float)
    """
    # YOUR CODE GOES HERE
    loss=0.0
    for i in range(x.shape[0]):
        factor=x[i].dot(w)
        1 if factor >= 0 else -1 
        if (factor != y[i]):
            loss+=np.log(1+np.exp(-y[i]*(x[i].dot(w))))
    return (loss/x.shape[0])

def sgd(pnts, nepochs=50, alpha=0.025, w0=(0.3, 0.3, 0.3)):
    """
    Update your parameters using SGD for nepochs
    @param  pnts: all x and y points concatentated together (NumPy array)
    @param  nepochs: number of epochs to train (integer)
    @param  alpha: learning rate (float)
    @param  w0: initial weights (tuple)
    @return w: The final weights (NumPy array)
    @return loss_arr: An array contains the loss at each epoch (list)
    """
    w = np.asarray(w0)
    loss_arr = []
    for e in range(nepochs):
        np.random.seed(SEED)
        np.random.shuffle(pnts)
        for data in range(pnts.shape[0]):
            w-=alpha*gradient_loss(pnts[data,3], pnts[data,:3], w)
        loss_arr.append(loss_fn(pnts[:,3], pnts[:,:3], w))
    return w, loss_arr

def sgd_nesterov(pnts, nepochs=50, alpha=0.025, w0=(0.3, 0.3, 0.3), momentum=0.9):
    """
    Update your parameters using Nesterov Accelerated Gradient for nepochs
    @param  pnts: all x and y points concatentated together (NumPy array)
    @param  nepochs: number of epochs to train (integer)
    @param  alpha: learning rate (float)
    @param  w0: initial weights (tuple)
    @param  momentum: nesterov momentum (float)
    @return w: The final weights (NumPy array)
    @return loss_arr: An array contains the loss at each epoch (list)
    """
    w = np.asarray(w0)
    v = np.zeros(w.shape)
    w_prev=w0
    v_prev=np.zeros(w.shape)
    loss_arr = []
    for e in range(nepochs):
        np.random.seed(SEED)
        np.random.shuffle(pnts)
        for data in range(pnts.shape[0]):
            w=w_prev-momentum*v_prev
            v=momentum*v_prev+alpha*gradient_loss(pnts[data,3], pnts[data,:3], w)
            w=w_prev-v   
            w_prev=np.copy(w)
            v_prev=np.copy(v)
        loss_arr.append(loss_fn(pnts[:,3], pnts[:,:3], w))
    return w, loss_arr

def adam(pnts, nepochs=50, alpha=0.025, w0=(0.3, 0.3, 0.3), betas=(0.9, 0.999), epsilon=1e-8):
    """
    Update your parameters using Adam for nepochs
    @param  pnts: all x and y points concatentated together (NumPy array)
    @param  nepochs: number of epochs to train (integer)
    @param  alpha: learning rate (float)
    @param  w0: initial weights (tuple)
    @param  betas: beta1 and beta2 values (tuple)
    @param  epsilon: epsilon to avoid divide by zero (float)
    @return w: The final weights (NumPy array)
    @return loss_arr: An array contains the loss at each epoch (list)
    """
    w = np.asarray(w0)
    m = np.zeros(w.shape)
    v = np.zeros(w.shape)
    w_prev=w0
    v_prev=np.zeros(w.shape)
    m_prev=np.zeros(w.shape)
    loss_arr = []
    for e in range(nepochs):
        np.random.seed(SEED)
        np.random.shuffle(pnts)
        for data in range(pnts.shape[0]):
            m=betas[0]*m_prev+(1-betas[0])*gradient_loss(pnts[data,3], pnts[data,:3], w_prev)
            v=betas[1]*v_prev+(1-betas[1])*(gradient_loss(pnts[data,3], pnts[data,:3], w_prev))**2
            m_new=m/(1-betas[0])
            v_new=v/(1-betas[1])
            w=w_prev-(alpha*m_new/(np.sqrt(v_new)+epsilon))
            w_prev=np.copy(w)
            v_prev=np.copy(v)
            m_prev=np.copy(m)
        loss_arr.append(loss_fn(pnts[:,3], pnts[:,:3], w))
    return w, loss_arr

# this is sanity check code for your convience
init_grad = gradient_loss(data_points[0,3], data_points[0,:3], np.array([0.3, 0.3, 0.3]))
if (np.allclose(init_grad, sanity_check['grad'], atol=TOL)):
    print('Sanity check - Gradient calculated matching' + u'\u2713')
else:
    print('Sanity check - Gradient calculated isn\'t matching :-(')

init_loss = loss_fn(data_points[:,3], data_points[:,:3], np.array([0.3, 0.3, 0.3]))
if (np.allclose(init_loss, sanity_check['loss'], atol=TOL)):
    print('Sanity check - Loss calculated matching' + u'\u2713')
else:
    print('Sanity check - Loss calculated isn\'t matching :-(')

sgd_weights, sgd_losses = sgd(data_points, nepochs=50, alpha=0.025, w0=(0.3, 0.3, 0.3))
if (np.allclose(sgd_weights, sanity_check['sgd'], atol=TOL)):
    print('Sanity check - SGD weights correctly match' + u'\u2713')
else:
    print('Sanity check - SGD weights aren\'t matching :-(')

nesterov_weights, nesterov_losses = sgd_nesterov(data_points, nepochs=50, alpha=0.025, w0=(0.3, 0.3, 0.3), momentum=0.9)
if (np.allclose(nesterov_weights, sanity_check['nesterov'], atol=TOL)):
    print('Sanity check - SGD Nesterov weights correctly match' + u'\u2713')
else:
    print('Sanity check - SGD Nesterov weights aren\'t matching :-(')

adam_weights, adam_losses = adam(data_points, nepochs=50, alpha=0.025, w0=(0.3, 0.3, 0.3), betas=(0.9, 0.999), epsilon=1e-8)
if (np.allclose(adam_weights, sanity_check['adam'], atol=TOL)):
    print('Sanity check - Adam weights correctly match' + u'\u2713')
else:
    print('Sanity check - Adam weights aren\'t matching :-(')

# DO NOT REMOVE THIS CODE
# we will use this code to see your graph to verify it makes sense.
# Loss plot
plt.figure()
plt.plot(sgd_losses, '-r', label='SGD')
plt.plot(nesterov_losses, '-g', label='SGD Nesterov')
plt.plot(adam_losses, '-b', label='Adam')
plt.title('Loss vs epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Decision boundary plot
x = np.linspace(-3,3,100)
plt.figure()
plt.scatter(data_points[:,0], data_points[:,1], c=data_points[:,3])
plt.plot(x, -(sgd_weights[0]/sgd_weights[1])*x - (sgd_weights[2]/sgd_weights[1]), '-r', label='SGD')
plt.plot(x, -(nesterov_weights[0]/nesterov_weights[1])*x - (nesterov_weights[2]/nesterov_weights[1]), '-g', label='SGD Nesterov')
plt.plot(x, -(adam_weights[0]/adam_weights[1])*x - (adam_weights[2]/adam_weights[1]), '-b', label='Adam')
plt.title('Decision Boundaries')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.show()