# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:15:17 2020

@author: Punit
"""


import numpy as np

def e_step(x, pi, mu, sigma):
    """
    @brief      Fix model parameters, update assignments
    @param      x      Samples represented as a numpy array of shape (n, d)
    @param      pi     Mixing coefficient represented as a numpy array of shape
                       (k,)
    @param      mu     Mean of each distribution represented as a numpy array of
                       shape (k, d)
    @param      sigma  Covariance matrix of each distribution represented as a
                       numpy array of shape (k, d, d)
    @return     The "soft" assignment of shape (n, k)
    """
    # IMPLEMENT ME
    pass


def m_step(x, a):
    """
    @brief      Fix assignments, update parameters
    @param      x     Samples represented as a numpy array of shape (n, d)
    @param      a     Soft assignments of each sample, represented as a numpy
                      array of shape (n, k)
    @return     A tuple (pi, mu, sigma), where
                - pi is the mixing coefficient represented as a numpy array of
                shape (k,)
                - mu is the mean of each distribution represented as a numpy
                array of shape (k, d)
                - sigma is the covariance matrix of each distribution
                represented as a numpy array of shape (k, d, d)
    """
    # IMPLEMENT ME
    pass


# # UNCOMMENT ME
# # run for a bunch of iterations
# iters = 30
# for i in range(iters):
#     # IMPLEMENT ME
#     pass

# # plot the learned distribution
# scatter_plot(a)
    

import numpy as np
from scipy.stats import multivariate_normal

def e_step(x, pi, mu, sigma):
    """
    @brief      Fix model parameters, update assignments
    @param      x      Samples represented as a numpy array of shape (n, d)
    @param      pi     Mixing coefficient represented as a numpy array of shape
                       (k,)
    @param      mu     Mean of each distribution represented as a numpy array of
                       shape (k, d)
    @param      sigma  Covariance matrix of each distribution represented as a
                       numpy array of shape (k, d, d)
    @return     The "soft" assignment of shape (n, k)
    """
    # IMPLEMENT ME
    rows=x.shape[0]
    cols=pi.shape[0]
    A=np.zeros((rows,cols))
    #print(pi.shape)
    #print(pi)
    #print(A)
    #print("data shape",x.shape)
    #print(mu)
    #print(sigma)
    #print(sigma[0])
    #print(sigma[0,:,:])
    for r in range(rows): # no of data
        deno=0.0
        for c in range(cols): # no of clusters
            numo=multivariate_normal.pdf(x[r,:],mean=mu[c,:],cov=sigma[c,:,:] )*pi[c]
            A[r,c]=numo
            deno+=numo
        A[r,:]/=deno    
    return A


def m_step(x, a):
    """
    @brief      Fix assignments, update parameters
    @param      x     Samples represented as a numpy array of shape (n, d)
    @param      a     Soft assignments of each sample, represented as a numpy
                      array of shape (n, k)
    @return     A tuple (pi, mu, sigma), where
                - pi is the mixing coefficient represented as a numpy array of
                shape (k,)
                - mu is the mean of each distribution represented as a numpy
                array of shape (k, d)
                - sigma is the covariance matrix of each distribution
                represented as a numpy array of shape (k, d, d)
    """
    # IMPLEMENT ME
    #print("mu in m step",mu)
    rows=x.shape[0] # no. of data points is the rows of x
    cols=x.shape[1] # no. of dimensions
    a_cols=a.shape[1] # no. of clusters is the cols of a
    pi=np.sum(a,axis=0)/rows
    mu=np.zeros((a_cols,cols))
    sigma=np.zeros((a_cols,cols,cols))
    for k in range(a_cols): #no. of clusters
        tmp_mu = np.zeros((cols))  # no. of dimensions
        for d in range(rows): # no of data points
            tmp_mu += (x[d, :] * a[d, k])
        mu[k] =tmp_mu/(pi[k]*rows)
    for k in range(a_cols): #no. of clusters
        tmp_sigma = np.zeros((cols,cols))  # no. of dimensions
        for d in range(rows): # no of dimensions in x -- d
            tmp_sigma += a[d, k]* np.outer((x[d, :] - mu[k, :]),(x[d, :] - mu[k, :]))
        sigma[k,:,:] = tmp_sigma/(pi[k]*rows)#(rows*deno)
    #print(sigma)
    return (pi, mu, sigma)


# # UNCOMMENT ME
# # run for a bunch of iterations
iters = 30
for i in range(iters):
    a=e_step(x, pi, mu, sigma)
    pi, mu, sigma=m_step(x, a)

# # plot the learned distribution
scatter_plot(a)