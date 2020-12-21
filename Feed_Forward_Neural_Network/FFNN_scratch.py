# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 00:11:01 2020

@author: Punit
"""


# This is the code that is used to set up the problem and produce test data for your submission. 
# It is reproduced here for your information, or if you would like to run your submission outside of . 
# You should not copy/paste this code into the code box below. This code is run automatically 'behind the scenes' before
#  your submitted code. 

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

data = np.loadtxt(BytesIO(data_files["data/wine/wine.data"]), dtype=float, delimiter=',')
overall_data =  data[:,1:]
overall_labels = data[:,0].astype(int)

max_array = np.amax(overall_data, axis = 0)
min_array = np.amin(overall_data, axis = 0)
denominator = max_array - min_array
num_data = len(overall_labels)
for i in range(num_data):
    overall_data[i] = (overall_data[i]-min_array)/denominator

indices = np.ones(len(overall_labels),dtype= bool)
indices[10:21] = False
indices[60:71] = False
indices[131:141] = False

training_data = overall_data[indices]
training_label = overall_labels[indices]
reversed_indices = np.logical_not(indices)
testing_data = overall_data[reversed_indices]
testing_label = overall_labels[reversed_indices]

#shuffle
perm = prng.permutation(len(training_label))
training_data = training_data[perm]
training_label = training_label[perm]

def softmax(x):
    """
    A numerically stable version of the softmax function
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def to_one_hot(y, k):
    """
    @brief      Convert numeric class values to one hot vectors
    @param      y     An array of labels of length N
    @param      k     Number of classes
    @return     A 2d array of shape (N x K), where K is the number of classes
    """
    n = y.shape[0]
    one_hot = np.zeros((n, k))
    one_hot[np.arange(n), y - 1] = 1
    return one_hot

def accuracy(y, y_hat):
    """
    @param      y      ground truth labels of shape (N x K)
    @param      y_hat  Estimated probability distributions of shape (N x K)
    @return     the accuracy of the prediction as a scalar
    """
    return (np.argmax(y, axis=1) == np.argmax(y_hat, axis=1)).mean()

def clear_grad(model):
    """
    clear the gradient in the parameters and replace them with 0's
    """
    for name, param, grad in model.named_parameters():
        name_split = name.split(".")
        child_name = name_split[0]
        param_name = name_split[1]
        model.children[child_name].grads[param_name] = np.zeros_like(model.children[child_name].grads[param_name])

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


import numpy as np
import matplotlib.pyplot as plt

def weight_init(fan_in, fan_out):
    """
    @param      fan_in   The number of input units
    @param      fan_out  The number of output units
    @return     The 2d weight matrix initialized using xavier uniform initializer
    """
    # IMPLEMENT ME
    #print(fan_in, fan_out)
    limits=np.sqrt(6/(fan_in+fan_out))
    #print("fan in is=",fan_in)
    #print("fan out is=", fan_out)
    return  np.random.uniform(-limits,limits,(fan_out,fan_in))

class ReLU(Module):
    def __init__(self):
        super().__init__()
        
        

    def forward(self, x):
        """
        @brief      Takes a batch of input and compute the ReLU output
        @param      x     A numpy array as input (N, in_features)
        @return     The output at the ReLU layer (N, in_features)
        """
        # IMPLEMENT ME
        #print(x)
        self.relu_forward=np.copy(x)
        return x* (x > 0)

    def backward(self, g):
        """
        @brief      Compute the gradients for parameters
        @param      g     The gradient of previous layers
        @return     The gradients of the loss w.r.t the input of this layer
        """
        #print(g)
        
        self.relu_forward[self.relu_forward<=0]=0
        self.relu_forward[self.relu_forward>0]=1
        return g*self.relu_forward
         

class Linear(Module):
    def __init__(self, weight, bias):
        super().__init__()
        self._register_param('weight', weight)
        self._register_param('bias', bias)
        

    def forward(self, x):
        """
        @brief      Takes a batch of input and compute the linear output
        @param      x     A numpy array as input (N, in_features)
        @return     The output of the linear layer (N, out_features)
        """
        # IMPLEMENT ME
        self.input_linear=np.copy(x)
        #print("shape of input x and weight in the linear",x.shape,self.params['weight'].shape)
        #print("this is w in linear \n",self.params['weight'] )
        self.lin_forward=x.dot(self.params['weight'].T)+self.params['bias']
        #print("shape of lin forward",self.lin_forward.shape)
        return self.lin_forward

    def backward(self, g):
        """
        @brief      Compute the gradients for parameters
        @param      g     The gradient of previous layers (N, out_features)
        @return     The gradients of the loss w.r.t the input of this layer (N, in_features)
        """
        #IMPLEMENT ME
        #relu_forward
        #self.lin_forward.dot(g.T)
        
        #print("g shape in linear",g.shape)
        #print("input x shape in linear", self.input_linear.shape)
        lin_grad=g.dot(self.params['weight'])#+ np.sum(g, axis=0)
        self.grads['weight']=self.grads['weight']+g.T.dot(self.input_linear)
        self.grads['bias']=self.grads['bias']+np.sum(g, axis=0)
        return lin_grad

class NeuralNetwork(Module):
    def __init__(self, d, h, k):
        """
        @brief      Initialize weight and bias
        @param      d     size of the input layer
        @param      h     size of the hidden layer
        @param      k     size of the output layer
        """
        super().__init__()
        wb = weight_init(d + 1, h)
        w1 = wb[:, :d]
        #print("this is the shape of w1",w1.shape)
        #print("this is w1 \n",w1)
        b1 = wb[:, d]
        #print("hidden, initial, final layer ",h,d,k)
        wb = weight_init(h + 1, k)
        w2 = wb[:, :h]
        #print("this is the shape of w2",w2.shape)
        #print("this is w2 \n",w2)
        b2 = wb[:, h]
        self._register_child('Linear1', Linear(w1, b1))
        self._register_child('ReLU', ReLU())
        self._register_child('Linear2', Linear(w2, b2))

    def forward(self, x):
        #Linear(self.params['Linear1'],self.params['Linear1'])
        
        #ReLU()
        #Linear2()
        """
        @brief      Takes a batch of samples and compute the feedforward output
        @param      x     A numpy array of shape (N x D)
        @return     The output at the last layer (N x K)
        """
        #IMPLEMENT ME
        
        self.input_data_shape=x.shape
        L1=self.children["Linear1"]
        out_linear1=L1.forward(x)
        #print("after linear 1 \n",out_linear1 )
        #print("shape output of 1 linear layer is ",out_linear1.shape)
        relu_1=self.children["ReLU"]
        out_relu=relu_1.forward(out_linear1)
        #print("shape output of ReLU layer is \n ",out_relu)
        L2=self.children["Linear2"]
        out_linear2=L2.forward(out_relu)
        #print("after L2",out_linear2)
        out_forward=np.zeros((out_linear2.shape))
        #print("self.children bias shape\n",self.children["Linear1"].grads["weight"])
        return out_linear2
        

    def backward(self, y, y_hat):
        """
        @brief      Compute the gradients for Ws and bs, you don't
                    need to return anything
        @param      y      ground truth label of shape (N x K)
        @param      y_hat  predictions of shape (N x K)
        """
        #IMPLEMENT ME
        
        #print(y.shape,y_hat.shape)
        soft_grad=y_hat-y
        #print("this is soft_grad \n",soft_grad)
        L2=self.children["Linear2"]
        out_linear2=L2.backward(soft_grad)
        relu_2=self.children["ReLU"]
        out_relu2=relu_2.backward(out_linear2)
        L1=self.children["Linear1"]
        out_linear1=L1.backward(out_relu2)

def update_param(model, lr):
    """
    update the parameters of the network
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #print(lr)
    #print("printing all the weigts of the model \n",model.children['Linear1'].grads['weight'])
    #print("printing all the weigts of the model \n",model.params['weight'])
    #my_module.params['weight']
    #print("input data shape", model.input_data_shape)
    #print("printing all the weigts of the model \n",model.children['Linear1'].params['weight'].shape)
    model.children['Linear1'].params['weight']-=(lr/model.input_data_shape[0])*model.children['Linear1'].grads['weight']
    model.children['Linear1'].params['bias']-=(lr/model.input_data_shape[0])*model.children['Linear1'].grads['bias']
    model.children['Linear2'].params['weight']-=(lr/model.input_data_shape[0])*model.children['Linear2'].grads['weight']
    model.children['Linear2'].params['bias']-=(lr/model.input_data_shape[0])*model.children['Linear2'].grads['bias']
    

def train_one_epoch(model, x, y, test_x, test_y, lr):
    """
    @brief      Takes in a model and train it for one epoch.
    @param      model   The neural network
    @param      x       The features of training data (N x D)
    @param      y       The labels of training data (N x K)
    @param      test_x  The features of testing data (M x D)
    @param      test_y  The labels of testing data (M x K)
    @param      lr      Learning rate
    @return     (train_accuracy, test_accuracy), the training accuracy and
                testing accuracy of the current epoch
    """
    # IMPLEMENT ME
    clear_grad(model)
    train_accuracy=0.0
    test_accuracy=0.0
    #for epochs in range(10):
        #for batch in range(x.shape[0] // batch_size):
    #batch_indices = np.random.choice(range(x.shape[0]), size=batch_size)
    # Create a mini-batch of training data and labels
    #X_batch = x[batch_indices]
    #y_batch = y[batch_indices]
    #print("shape of training data",x.shape)
    out_linear2=model.forward(x)
    #print("this is model output shape \n",out_linear2.shape)
    y_hat=np.zeros((out_linear2.shape))
    for out in range(out_linear2.shape[0]):
        y_hat[out]=softmax(out_linear2[out])
    #print("softmax shape \n",y_hat.shape)
    #print("this is y shape",y.shape)
    train_accuracy=accuracy(y, y_hat)
    #print("train accuracy is",train_accuracy)
    model.backward(y, y_hat)
    #print("after coming out of back prop",out_back.shape)
    #print("printing shape in trainign \n",model.children['Linear2'].grads['bias'].shape)
    update_param(model, lr)
    test=model.forward(test_x)
    y_hat_test=np.zeros((test.shape))
    for out in range(test.shape[0]):
         y_hat_test[out]=softmax(test[out])
    test_accuracy=accuracy(test_y, y_hat_test)
    #print("test accuracy is",test_accuracy)
    return (train_accuracy, test_accuracy)

# Implement step 6 here
y = to_one_hot(training_label, 3)
test_y = to_one_hot(testing_label, 3)
my_model=NeuralNetwork(13, 50, 3 )
train_accuracy=np.zeros((100))
test_accuracy=np.zeros((100))
for epoch in range(100):
    train_accuracy[epoch], test_accuracy[epoch]=train_one_epoch(my_model, training_data , y , testing_data , test_y , 0.3)
plt.plot(train_accuracy , label='train accuracy ')
plt.plot(test_accuracy , label='test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy ')
plt.legend()
print(" For the different learning parameter values I observed the following \n \
      1. The value of  0.03  is makes the gradeint steps too low and given the fixed number of epochs the gradient does not converge \n \
      2. The value of  3.0 is too high and instead of converging to optimum values our model overshoots it and ocscillates around the minima \n \
      3. The value of 0.3 is the optimum values and gives us the testing accuracy of almost 100% as the number of epochs increases ")
nn_model = my_model
            
            
            
            
            
            
