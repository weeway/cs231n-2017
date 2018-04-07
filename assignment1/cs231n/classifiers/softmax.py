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
  num_train = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = X[i,:].dot(W)
    idx_max = np.argmax(scores)
    scores -= np.max(scores)
    scores = np.exp(scores)
    p = scores/np.sum(scores)
    loss -= np.log(p[y[i]])
    
    for j in range(num_class):
        if y[i] == j:
            dW[:,j] += (p[j]-1)*X[i]
        else:
            dW[:,j] += p[j]*X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= num_train
  loss += reg*np.sum(W*W)
  dW /= num_train
  dW += 2*reg*W
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
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores,axis=1).reshape(-1,1)
  scores = np.exp(scores)
  p = scores/np.sum(scores,axis=1).reshape(-1,1)
  idx_correct = [[x for x in range(num_train)], y]
  scores_correct = p[idx_correct]
  loss = np.sum(-np.log(scores_correct))
  
  loss /= num_train
  loss += reg*np.sum(W*W)

  p[idx_correct] -= 1
    
  dW = X.T.dot(p)
  dW /= num_train
  dW += 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

