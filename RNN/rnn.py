# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:35:58 2020

@author: Punit
"""


import numpy as np
import time
from io import BytesIO
import time
ten_sec = int(round(time.time()/20))
#314, 10,11
if ten_sec %3 == 0:
    prng = np.random.RandomState(314)
elif ten_sec%3 == 1:
    prng = np.random.RandomState(11)
else:
    prng = np.random.RandomState(10)

def to_one_hot(y, k):
    (n, t) = y.shape
    one_hot = np.zeros((n, t, k))
    for sample in range(n):
        one_hot[sample, np.arange(t), y[sample]] = 1
    return one_hot

def softmax(x):
    """
    A numerically stable version of the softmax function
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

class Module:
    def __init__(self):
        super().__init__()
        self.params = dict()
        self.grads = dict()
        self.children = dict()
        self.cache = dict()

    def _register_param(self, name: str, param: np.ndarray):
        """ the parameter can be accessed via self.params[name]
        the gradient can be accessed via self.grads[name]
        """
        assert isinstance(param, np.ndarray)
        self.params[name] = param
        self.grads[name] = np.zeros_like(param)

    def _register_child(self, name: str, child: 'Module'):
        """ the module can be acccessed via self.children[name]"""
        assert isinstance(child, Module)
        self.children[name] = child

    def forward(self, *x):
        raise NotImplementedError

    def backward(self, *g):
        raise NotImplementedError

    def named_parameters(self, base: tuple = ()):
        """recursively get all params in a generator"""
        assert self.params.keys() == self.grads.keys()
        for name in self.params:
            full_name = '.'.join(base + (name,))
            yield (full_name, self.params[name], self.grads[name])

        # recursively on others
        for child_name, child in self.children.items():
            yield from child.named_parameters(base=base + (child_name,))

def weight_init(fan_in, fan_out):

    a = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(low=-a, high=a, size=(fan_out, fan_in))

def clear_grad(model):
    """
    clear the gradient in the parameters and replace them with 0's
    """
    for name, param, grad in model.named_parameters():
        name_split = name.split(".")
        child_name = name_split[0]
        param_name = name_split[1]
        model.children[child_name].grads[param_name] = np.zeros_like(model.children[child_name].grads[param_name])

def update_param(model, lr):
    """
    update the parameters of the network
    """
    for name, param, grad in model.named_parameters():
        name_split = name.split(".")
        child_name = name_split[0]
        param_name = name_split[1]
        model.children[child_name].params[param_name] -= lr * model.children[child_name].grads[param_name]

data_file = "data/hw5_language_model.txt"
data = np.loadtxt(BytesIO(data_files[data_file]), dtype=np.int, delimiter=',')
X_all = to_one_hot(data, 4)
X = X_all[30:]
test_X = X_all[:30]


import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

#Ref--- I have taken this function from HW2 problem set up code part.
#NetIDs :andrew24,mdugar2,mmallon3,seantm3,tbeirne2,viraj2,mjedh2

def softmax_batch(z):
    y_hat = np.empty_like(z)
    for i in range(z.shape[0]):
        y_hat[i] = softmax(z[i])
    return y_hat

def cross_entropy(Y, Y_hat):
    """
    @brief      Compute cross-entropy loss between labels and predictions
                averaged across the N instances
    @param      Y       ground-truth label of shape (N x T x K)
    @param      Y_hat   predictions of shape (N x T x K)
    @return     the average cross-entropy loss between Y and Y_hat
    """
    # IMPLEMENT ME
    
    #print("predictions truth",Y_hat)
    loss=0.0
    N,T,K=Y.shape
    for n in range(0,N):
        loss+=sum(sum((-Y[n])*(np.log(Y_hat[n]+1e-9))))
    #print(loss)
    return loss/N

def generate_labels(X):
    """
    @brief      Takes in samples and generates labels.
    @param      X       Samples of sequence data (N x T x D)
    @return     Y, labels of shape (N x T x K)
    """
    # IMPLEMENT ME
    N,T,D=X.shape
    Y=np.zeros((N,T+1,D+1))
    Y[:,:-1,:-1] = X
    Y = numpy.delete(Y, 0, axis=1)
    Y[:,-1,-1]=1
    return Y

class RNNCell(Module):
    def __init__(self, parameters):
        super().__init__()
        self._register_param('V', parameters['V'])
        self._register_param('W', parameters['W'])
        self._register_param('U', parameters['U'])
        self._register_param('c', parameters['c'])
        self._register_param('b', parameters['b'])

    def forward(self, x_t, h_prev):
        """
        @brief      Takes a batch of input at the timestep t with the previous hidden states and compute the RNNCell output
        @param      x_t    A numpy array as the input (N, in_features)
        @param      h_prev A numpy array as the previous hidden state (N, hidden_features)
        @return     The current hidden state (N, hidden_features) and the prediction at the timestep t (N, out_features)
        """
        # IMPLEMENT ME

        #print("shape of w",self.params['W'].shape)
        #print("shape of b",self.params['b'].shape)
        #print("shape of u",self.params['U'].shape)
        #print("shape of v",self.params['V'].shape)
        #print("shape of c",self.params['c'].shape)
        #print("input",x_t.shape)
        #print("hidden",h_prev.shape)
        
        #################################################################
        h_loop=np.zeros(h_prev.shape)
        for nn in range(0, x_t.shape[0]):
            h_loop[nn]=np.tanh(np.dot(self.params['W'],h_prev[nn])+ np.dot(self.params['U'],x_t[nn] )+self.params['b'])
        #print("h loop",h_loop) #h is 3x6
        y_lol=np.zeros((h_loop.shape[0],self.params['V'].shape[0])) #3x5
        for nn in range(0, h_loop.shape[0]):
            y_lol[nn]=(np.dot(h_loop[nn],self.params['V'].T)+self.params['c'])
        y_lol_new=np.zeros(y_lol.shape)
        for row in range(0,y_lol.shape[0]):
            y_lol_new[row]=softmax(y_lol[row])
        #################################################################
        
        return h_loop,y_lol_new

    def backward(self, x_t, y_t, y_hat_t, dh_next, h_t, h_prev):
        """
        @brief      Compute and update the gradients for parameters of RNNCell at the timestep t
        @param      x_t      A numpy array as the input (N, in_features)
        @param      y_t      A numpy array as the target (N, out_features)
        @param      y_hat_t  A numpy array as the prediction (N, out_features)
        @param      dh_next  A numpy array as the gradient of the next hidden state (N, hidden_features)
        @param      h_t      A numpy array as the current hidden state (N, hidden_features)
        @param      h_prev   A numpy array as the previous hidden state (N, hidden_features)
        @return     The gradient of the current hidden state (N, hidden_features)
        """
        diff=y_hat_t-y_t
        self.d_h= (np.dot(diff,self.params['V'])+ np.dot(dh_next,self.params['W'])) #from Piazza
        self.grads['V']+=np.dot(h_t.T,diff).T #from Piazza
        self.grads['U']+=np.dot(((1-h_t**2)*self.d_h).T,x_t) #from Piazza
        self.grads['W']+=np.dot(((1-h_t**2)*self.d_h).T,h_prev) #from Piazza
        #print(self.grads['W'])
        self.grads['c']+=sum(diff)
        self.grads['b']+= sum((1-h_t**2)*self.d_h) 
        self.current=self.d_h*(1-h_t**2) # after advice from TA
        return self.current

class RNN(Module):
    def __init__(self, d, h, k):
        """
        @brief      Initialize weight and bias
        @param      d   size of the input layer
        @param      h   size of the hidden layer
        @param      k   size of the output layer
        NOTE: Do not change this function or variable names; they are
            used for grading.
        """
        super().__init__()
        self.d = d
        self.h = h
        self.k = k

        parameters = {}
        wb = weight_init(d + h + 1, h)
        parameters['W'] = wb[:, :h]
        parameters['U'] = wb[:, h:h+d]
        parameters['b'] = wb[:, h+d]
        
        wb = weight_init(h + 1, k)
        parameters['V'] = wb[:, :h]
        parameters['c'] = wb[:, h]
        self._register_child('RNNCell', RNNCell(parameters))
    
    def forward(self, X):
        """
        @brief      Takes a batch of samples and computes the RNN output
        @param      X   A numpy array as the input of shape (N x T x D)
        @return     Hidden states (N x T x H), RNN's output (N x T x K)
        """
        # IMPLEMENT ME
        #print(X.shape)
        #print(X.shape)
        N,T,D=X.shape
        hidden=np.zeros((N,T,self.h))
        self.output=np.zeros((N,T,self.k))
        for x in range(T):
            hidden[:,x,:], self.output[:,x,:] = self.children['RNNCell'].forward(X[:,x,:], hidden[:,x-1,:])
        
        return  hidden,self.output

    def backward(self, X, Y, Y_hat, H):
        """
        @brief      Backpropagation of the RNN
        @param      X      A numpy array as the input of shape (N x T x D)
        @param      Y      A numpy array as the ground truth labels of shape (N x T x K)
        @param      Y_hat  A numpy array as the prediction of shape (N x T x K)
        @param      H      A numpy array as the hidden states after the forward of shape (N x T x H)
        """
        # IMPLEMENT ME
        #(self, x_t, y_t, y_hat_t, dh_next, h_t, h_prev)
        #print(self.grads['h'])
        N,T,D=X.shape
        dh_next=np.zeros((N,self.h))
        
        for x in range(T-1,-1,-1):
            if(x != 0):
                dh_next = self.children['RNNCell'].backward(X[:,x,:],Y[:,x,:],Y_hat[:,x,:],dh_next,H[:,x,:],H[:,x-1,:] )
            else:
                h_prev=np.zeros((N,self.h))
                dh_next = self.children['RNNCell'].backward(X[:,x,:],Y[:,x,:],Y_hat[:,x,:],dh_next,H[:,x,:],h_prev)
        
        
        

def train_one_epoch(model, X, test_X, lr):
    """
    @brief      Takes in a model and train it for one epoch.
    @param      model   The recurrent neural network
    @param      X       The features of training data (N x T x D)
    @param      test_X  The features of testing data (M x T x D)
    @param      lr      Learning rate
    @return     (train_cross_entropy, test_cross_entropy), the cross
                entropy loss for train and test data
    """
    # IMPLEMENT ME
    clear_grad(model)
    Y=generate_labels(X)
    Y_test=generate_labels(test_X)
    hidden,output=model.forward(X)
    Y_hat = softmax_batch(output)
    train_cross_entropy=cross_entropy(Y, Y_hat)
    model.backward(X, Y, output, hidden)
    #print("batch size",X.shape[0])
    #print("batch size",X)
    discount=lr/X.shape[0]
    update_param(model,discount)
    hidden_test,Y_hat_test=model.forward(test_X)
    Y_hat_test = softmax_batch(Y_hat_test)
    test_cross_entropy=cross_entropy(Y_test, Y_hat_test)
    return (train_cross_entropy, test_cross_entropy)

d = 4
h = 5
lr = 0.05
#model = RNN(d, h, d+1)
#train_cross_entropy, test_cross_entropy=train_one_epoch(model, X, test_X, lr)
#print(train_cross_entropy, test_cross_entropy)
#print(X.shape)
#generate_labels(X)
#print(Y)



# import numpy as np
# import matplotlib.pyplot as plt

# def cross_entropy(Y, Y_hat):
#     """
#     @brief      Compute cross-entropy loss between labels and predictions
#                 averaged across the N instances
#     @param      Y       ground-truth label of shape (N x T x K)
#     @param      Y_hat   predictions of shape (N x T x K)
#     @return     the average cross-entropy loss between Y and Y_hat
#     """
#     (n, T, _) = Y.shape
#     total_loss = 0
#     for bi in range(n):
#         for t in range(T):
#             total_loss += -np.dot(Y[bi, t], np.log(Y_hat[bi, t]+1e-9))
#     return total_loss / n

# def generate_labels(X):
#     """
#     @brief      Takes in samples and generates labels.
#     @param      X       Samples of sequence data (N x T x D)
#     @return     Y, labels of shape (N x T x K)
#     """
#     (n, t, d) = X.shape
#     Y = np.zeros((n, t, d+1))
#     Y[:, :-1, :-1] = X[:, 1:]
#     Y[:, -1, d] = 1
#     return Y

# class RNNCell(Module):
#     def __init__(self, parameters):
#         super().__init__()
#         self._register_param('V', parameters['V'])
#         self._register_param('W', parameters['W'])
#         self._register_param('U', parameters['U'])
#         self._register_param('c', parameters['c'])
#         self._register_param('b', parameters['b'])

#     def forward(self, x_t, h_prev):
#         """
#         @brief      Takes a batch of input at the timestep t with the previous hidden states and compute the RNNCell output
#         @param      x_t    A numpy array as the input (N, in_features)
#         @param      h_prev A numpy array as the previous hidden state (N, hidden_features)
#         @return     The current hidden state (N, hidden_features) and the prediction at the timestep t (N, out_features)
#         """
#         h_next = np.tanh(np.dot(self.params['W'], h_prev.T) + np.dot(self.params['U'], x_t.T) + self.params['b'][:,None])
#         # h_next = np.dot(self.params['W'], h_prev.T) + np.dot(self.params['U'], x_t.T) + self.params['b'][:,None]
#         zt = np.dot(self.params['V'], h_next) + self.params['c'][:,None]
#         zt = zt.T
#         yt_hat = np.zeros_like(zt)
#         for i in range(zt.shape[0]):
#             yt_hat[i] = softmax(zt[i])

#         return h_next.T, yt_hat

#     def backward(self, x_t, y_t, y_hat_t, dh_next, h_t, h_prev):
#         """
#         @brief      Compute and update the gradients for parameters of RNNCell at the timestep t
#         @param      x_t      A numpy array as the input (N, in_features)
#         @param      y_t      A numpy array as the target (N, out_features)
#         @param      y_hat_t  A numpy array as the prediction (N, out_features)
#         @param      dh_next  A numpy array as the gradient w.r.t the next hidden state before the activation (N, hidden_features)
#         @param      h_t      A numpy array as the current hidden state (N, hidden_features)
#         @param      h_prev   A numpy array as the previous hidden state (N, hidden_features)
#         @return     The gradient w.r.t the current hidden state before the activation (N, hidden_features)
#         """
#         dL_dot = y_hat_t - y_t
#         self.grads['V'] += np.dot(dL_dot.T, h_t)
#         self.grads['c'] += np.sum(dL_dot, axis=0, keepdims=False)
#         dL_dht = np.dot(dh_next, self.params['W']) + np.dot(dL_dot, self.params['V']) # (n x h)
#         dL_dht = (1 -h_t**2)*dL_dht # (n x h)
#         self.grads['b'] += np.sum(dL_dht, axis=0, keepdims=False) # (n x h)
#         self.grads['U'] += np.dot(dL_dht.T, x_t)
#         self.grads['W'] += np.dot(dL_dht.T, h_prev)
        
#         return dL_dht

# class RNN(Module):
#     def __init__(self, d, h, k):
#         """
#         @brief      Initialize weight and bias
#         @param      d   size of the input layer
#         @param      h   size of the hidden layer
#         @param      k   size of the output layer
#         NOTE: Do not change this function or variable names; they are
#             used for grading.
#         """
#         super().__init__()
#         self.d = d
#         self.h = h
#         self.k = k

#         parameters = {}
#         wb = weight_init(d + h + 1, h)
#         parameters['W'] = wb[:, :h]
#         parameters['U'] = wb[:, h:h+d]
#         parameters['b'] = wb[:, h+d]
        
#         wb = weight_init(h + 1, k)
#         parameters['V'] = wb[:, :h]
#         parameters['c'] = wb[:, h]
#         self._register_child('RNNCell', RNNCell(parameters))
    
#     def forward(self, X):
#         """
#         @brief      Takes a batch of samples and computes the RNN output
#         @param      X   A numpy array as the input of shape (N x T x D)
#         @return     Hidden states (N x T x H), RNN's output (N x T x K)
#         """
#         (n, T, _) = X.shape
#         Y_hat = np.zeros((n, T, self.k))
#         H = np.zeros((n, T, self.h))
#         h_next = np.zeros((n, self.h))
#         for t in range(T):
#             h_next, y_hat_t = self.children['RNNCell'].forward(X[:,t,:], h_next)
#             Y_hat[:, t, :] = y_hat_t
#             H[:, t, :] = h_next
#         return H, Y_hat

#     def backward(self, X, Y, Y_hat, H):
#         """
#         @brief      Backpropagation of the RNN
#         @param      X      A numpy array as the input of shape (N x T x D)
#         @param      Y      A numpy array as the ground truth labels of shape (N x T x K)
#         @param      Y_hat  A numpy array as the prediction of shape (N x T x K)
#         @param      H      A numpy array as the hidden states after the forward of shape (N x T x H)
#         @return
#         """
#         (n, T, _) = X.shape
#         dL_dht = np.zeros(H[:,0,:].shape)
#         for t in range(T-1, -1, -1):
#             if t >= 1:
#                 h_next = H[:, t-1, :]
#             else:
#                 h_next = np.zeros_like(h_next)
#             dL_dht = self.children['RNNCell'].backward(X[:,t,:], Y[:,t,:], Y_hat[:,t,:], dL_dht, H[:,t,:], h_next)


# def train_one_epoch(model, X, test_X, lr):
#     """
#     @brief      Takes in a model and train it for one epoch.
#     @param      model   The recurrent neural network
#     @param      X       The features of training data (N x T x D)
#     @param      test_X  The features of testing data (M x T x D)
#     @param      lr      Learning rate
#     @return     (train_cross_entropy, test_cross_entropy), the cross
#                 entropy loss for train and test data
#     """
#     clear_grad(model)
#     # generate labels from data
#     Y = generate_labels(X)
#     test_Y = generate_labels(test_X)

#     # feed training data forward
#     H, Y_hat = model.forward(X)
#     train_cross_entropy = cross_entropy(Y, Y_hat)
#     model.backward(X, Y, Y_hat, H)
#     discount = lr / X.shape[0]
#     update_param(model, discount)

#     # feed testing data forward
#     _, test_Y_hat = model.forward(test_X)
#     test_cross_entropy = cross_entropy(test_Y, test_Y_hat)

#     return train_cross_entropy, test_cross_entropy

# d = 4
# h = 5
# lr = 0.05

# model = RNN(d, h, d+1)
# train_loss_list = []
# test_loss_list = []
# for i in range(100):
#     train_loss, test_loss = train_one_epoch(model, X, test_X, lr)
#     train_loss_list.append(train_loss)
#     test_loss_list.append(test_loss)



























