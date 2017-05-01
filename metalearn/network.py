"""
    network.py
    Nicholas S. Bradford
    30 April 2017

    Train a 2-layer neural network to classify images of hand-written digits from MNIST dataset.
    Implement gradient descent to minimize the cross-entropy loss function.
    Because there are 10 different outputs, there are 10 weights vectors;
        thus, the weight matrix is 784 x 10.

"""

import tensorflow as tf

def get_neural_dims(SIZE_INPUT, SIZE_OUTPUT, SIZE_H1):
    LEN_W1 = SIZE_INPUT * SIZE_H1
    LEN_B1 = SIZE_H1
    LEN_W2 = SIZE_H1 * SIZE_OUTPUT
    LEN_B2 = SIZE_OUTPUT
    NEURAL_DIMS = LEN_W1 + LEN_B1 + LEN_W2 + LEN_B2 
    return NEURAL_DIMS

NEURAL_DIMS = get_neural_dims(SIZE_INPUT=784, SIZE_OUTPUT=10, SIZE_H1=20)
NEURAL_FACE_DIMS = get_neural_dims(SIZE_INPUT=576, SIZE_OUTPUT=1, SIZE_H1=20)
NEURAL_DIMS_200UNITS = get_neural_dims(SIZE_INPUT=784, SIZE_OUTPUT=10, SIZE_H1=200)
# =================================================================================================
# Neural network problem 

def NEURAL_BASE(weights, batch, activation, SIZE_H1=20, SIZE_INPUT=784, SIZE_OUTPUT=10,
                        COST=tf.nn.softmax_cross_entropy_with_logits):
    """ This model has ~98.32% accuracy on MNIST.
        Args:
            weights (tf.Tensor): params of the optimizee network (weights and biases)
            weights = [W1, b1, W2, b2, W3, b3]
        Returns:
            (tf.Tensor): the cost (which we are trying to minimize) = 1/ % accurate on train set
    """
    with tf.variable_scope('Optimizee_ANN'):
        
        LEN_W1 = SIZE_INPUT * SIZE_H1
        LEN_B1 = SIZE_H1
        LEN_W2 = SIZE_H1 * SIZE_OUTPUT
        LEN_B2 = SIZE_OUTPUT
        DIMS = LEN_W1 + LEN_B1 + LEN_W2 + LEN_B2 

        batch_x, batch_y = batch[0], batch[1]
        W1 = tf.reshape(tf.slice(weights, [0], [LEN_W1]), [SIZE_INPUT, SIZE_H1], name='W1')
        b1 = tf.reshape(tf.slice(weights, [LEN_W1], [LEN_B1]), [SIZE_H1], name='b1')
        W2 = tf.reshape(tf.slice(weights, [LEN_W1+LEN_B1], [LEN_W2]), [SIZE_H1,SIZE_OUTPUT], name='W2')
        b2 = tf.reshape(tf.slice(weights, [LEN_W1+LEN_B1+LEN_W2], [LEN_B2]), [SIZE_OUTPUT], name='b1')
        h1 = activation(tf.matmul(batch_x, W1) + b1)
        yhat = tf.matmul(h1, W2) + b2
    with tf.variable_scope('Optimizee_Cost'):
        cross_entropy = tf.reduce_mean(COST(labels=batch_y, logits=yhat))
        # tf.scalar_summary('optimizee cross entropy', cross_entropy)
    return cross_entropy

    # correct_prediction = tf.equal(tf.argmax(yhat,1), tf.argmax(batch_y,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return cross_entropy #, accuracy


# tf.sigmoid, tf.tanh, tf.nn.elu, tf.

def f_neural(weights, batch):
    return NEURAL_BASE(weights, batch, activation=tf.nn.relu)

def f_neural_elu(weights, batch):
    return NEURAL_BASE(weights, batch, activation=tf.nn.elu)

def f_neural_sigmoid(weights, batch):
    return NEURAL_BASE(weights, batch, activation=tf.sigmoid)

def f_neural_tanh(weights, batch):
    return NEURAL_BASE(weights, batch, activation=tf.tanh)

# def f_neural_twohidden(weights, batch):
def f_neural_200units(weights, batch):
    return NEURAL_BASE(weights, batch, SIZE_H1=200, activation=tf.relu)

def f_neural_smile(weights, batch, activation):
    return NEURAL_BASE(weights, batch, activation=tf.nn.relu, SIZE_INPUT=576, SIZE_OUTPUT=1,
                        COST=tf.nn.sparse_softmax_cross_entropy_with_logits)


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
    scale = tf.random_uniform([10], 0.5, 1.5)
    x = scale * x
    return tf.reduce_sum(x*x)

# =================================================================================================

class Problem(object):
    def __init__(self, name, loss_fcn, DIMS, SIZE_INPUT=None):
        self.name = name
        self.loss_fcn = loss_fcn
        self.DIMS = DIMS
        self.SIZE_INPUT = SIZE_INPUT


PROBLEM_NEURAL = Problem('NeuralNet', f_neural, NEURAL_DIMS, SIZE_INPUT=784)
PROBLEM_QUADRATIC = Problem('Quadratic', f_quadratic, 10)
PROBLEM_NEURAL_ELU = Problem('NN_ELU', f_neural_elu, NEURAL_DIMS, SIZE_INPUT=784)
PROBLEM_NEURAL_SIGMOID = Problem('NN_SIGMOID', f_neural_sigmoid, NEURAL_DIMS, SIZE_INPUT=784)
PROBLEM_NEURAL_TANH = Problem('NN_TANH', f_neural_tanh, NEURAL_DIMS, SIZE_INPUT=784)
# PROBLEM_NEURAL_TWOHIDDEN = Problem('NN_TWOHIDDEN', f_neural_twohidden, NEURAL_DIMS_TWOHIDDEN)
PROBLEM_NEURAL_200UNITS = Problem('NN_200UNITS', f_neural_200units, NEURAL_DIMS_200UNITS)
PROBLEM_FACE = Problem('NN_FACE', f_neural_smile, NEURAL_FACE_DIMS, SIZE_INPUT=576)

NN_PROBLEMS = [
    PROBLEM_NEURAL,
    PROBLEM_NEURAL_ELU,
    PROBLEM_NEURAL_SIGMOID,
    PROBLEM_NEURAL_TANH,
    # PROBLEM_NEURAL_TWOHIDDEN,
    PROBLEM_NEURAL_200UNITS,
    PROBLEM_FACE
]
