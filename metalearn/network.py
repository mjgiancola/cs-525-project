"""
    network.py
    Nicholas S. Bradford
    30 April 2017

    Train a 2-layer neural network to classify images of hand-written digits from MNIST dataset.
    Implement gradient descent to minimize the cross-entropy loss function.
    Because there are 10 different outputs, there are 10 weights vectors;
        thus, the weight matrix is 784 x 10.

"""


SIZE_INPUT = 784
SIZE_OUTPUT = 10

import tensorflow as tf


# =================================================================================================
# Neural network problem 

def f_neural(weights, batch):
    """ This model has ~98.32% accuracy on MNIST.
        Args:
            weights (tf.Tensor): params of the optimizee network (weights and biases)
            weights = [W1, b1, W2, b2, W3, b3]
        Returns:
            (tf.Tensor): the cost (which we are trying to minimize) = 1/ % accurate on train set
    """
    SIZE_H1 = 200
    LEN_W1 = SIZE_INPUT * SIZE_H1
    LEN_B1 = SIZE_H1
    LEN_W2 = SIZE_H1 * SIZE_OUTPUT
    LEN_B2 = SIZE_OUTPUT
    DIMS = LEN_W1 + LEN_B1 + LEN_W2 + LEN_B2 

    batch_x, batch_y = batch[0], batch[1]
    W1 = tf.reshape(tf.slice(weights, [0], [LEN_W1]), [SIZE_INPUT, SIZE_H1])
    b1 = tf.reshape(tf.slice(weights, [LEN_W1], [LEN_B1]), [SIZE_H1])
    W2 = tf.reshape(tf.slice(weights, [LEN_W1+LEN_B1], [LEN_W2]), [SIZE_H1,SIZE_OUTPUT])
    b2 = tf.reshape(tf.slice(weights, [LEN_W1+LEN_B1+LEN_W2], [LEN_B2]), [SIZE_OUTPUT])
    h1 = tf.nn.relu(tf.matmul(batch_x, W1) + b1)
    yhat = tf.matmul(h1, W2) + b2
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_y, logits=yhat))
    return cross_entropy

    # correct_prediction = tf.equal(tf.argmax(yhat,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # accuracy_eval = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    # return 1 / accuracy_eval # we are returning a value to minimize


# =================================================================================================
# Multi-dimensional quadratic minimization problem

def f_quadratic(x, batch=None):
    """ Loss function. Represents the network we are trying to optimize, the "optimizee".
        Could also be thought of as the error landscape, or the "problem" we're solving.
        Args:
            x (tf.Tensor): params of the optimizee network (weights and biases)
        Returns:
            (tf.Tensor): the cost (which we are trying to minimize)
    """
    x = scale * x
    return tf.reduce_sum(x*x)

# =================================================================================================

class Problem(object):
    def __init__(self, name, loss_fcn, DIMS):
        self.name = name
        self.loss_fcn = loss_fcn
        self.DIMS = DIMS

SIZE_H1 = 200
LEN_W1 = SIZE_INPUT * SIZE_H1
LEN_B1 = SIZE_H1
LEN_W2 = SIZE_H1 * SIZE_OUTPUT
LEN_B2 = SIZE_OUTPUT
NEURAL_DIMS = LEN_W1 + LEN_B1 + LEN_W2 + LEN_B2 

PROBLEM_NEURAL = Problem('NeuralNet', f_neural, NEURAL_DIMS)
PROBLEM_QUADRATIC = Problem('Quadratic', f_quadratic, 10)
