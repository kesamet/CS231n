import numpy as np
from random import shuffle


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
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    f = X[i].dot(W)
    f -= np.max(f)
    probs = np.exp(f) / np.sum(np.exp(f))
    loss += -np.log(probs[y[i]])

    dscores = probs
    dscores[y[i]] -= 1
    dW += np.dot(X[i].reshape(-1,1), dscores.reshape(1,-1))
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += 2 * reg * W

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  # evaluate class scores
  f = np.dot(X,W)
    
  # compute the class probabilities
  f -= np.max(f, axis=1, keepdims=True)
  probs = np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_train),y])
  data_loss = np.sum(corect_logprobs) / num_train
  reg_loss = reg * np.sum(W * W)
  loss = data_loss + reg_loss
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_train),y] -= 1
  dscores /= num_train
  
  # backpropate the gradient to the parameter W
  dW = np.dot(X.T, dscores)
  dW += 2 * reg * W # regularization gradient
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

