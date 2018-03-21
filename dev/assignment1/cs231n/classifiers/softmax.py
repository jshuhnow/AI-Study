import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  f = X.dot(W)
  for i in range(num_train):
    f[i] -= np.max(f[i])
    
    exp_score = np.exp(f[i])
    exp_sum = np.sum(exp_score)
    
    loss += -np.log(exp_score[y[i]] / exp_sum)
    
    for j in range(num_class):
      
      if j==y[i]:
        dW[:,j] += (-1 + exp_score[j]/exp_sum) * X[i].T
      else:
        dW[:,j] += exp_score[j]/exp_sum * X[i].T
  
  loss /= num_train
  loss += reg * np.sum(W*W)
  dW /= num_train
  dW += 2.0*reg * W
      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  f = X.dot(W)
  f -= np.max(f, axis=1, keepdims = True)
  p = np.exp(f) / np.sum(np.exp(f), axis=1, keepdims = True)  
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = (np.sum(-np.log(p[np.arange(num_train),y])))/num_train + reg * np.sum(W*W)
  ind = np.zeros_like(p)
  ind[np.arange(num_train),y] = 1
  dW = X.T.dot(p - ind) / num_train + 2.0 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  
  return loss, dW

