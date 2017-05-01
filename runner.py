"""
    runner.py
    Nicholas S. Bradford
    30 April 2017

"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


# =================================================================================================
# Globals

# Import MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)




# DIMS = 10 # dimensionality of cost function space; equal to number of optimizee params
# scale = tf.random_uniform([DIMS], 0.5, 1.5)
TRAINING_STEPS = 20 # 100 in the paper ?

# LSTM params
NUM_LAYERS = 2
STATE_SIZE = 20 # n of hidden units

# =================================================================================================
# Neural network problem 

SIZE_INPUT = 784
SIZE_H1 = 200
SIZE_OUTPUT = 10

LEN_W1 = SIZE_INPUT * SIZE_H1
LEN_B1 = SIZE_H1
LEN_W2 = SIZE_H1 * SIZE_OUTPUT
LEN_B2 = SIZE_OUTPUT
DIMS = LEN_W1 + LEN_B1 + LEN_W2 + LEN_B2 

# TODO need to set these equal to the MNIST training batches
batch_x = tf.placeholder(tf.float32, [None, SIZE_INPUT], name='Batch')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name ='y')


def f_neural(weights):
    """ This model has ~98.32% accuracy on MNIST.
        Args:
            weights (tf.Tensor): params of the optimizee network (weights and biases)
            weights = [W1, b1, W2, b2, W3, b3]
        Returns:
            (tf.Tensor): the cost (which we are trying to minimize) = 1/ % accurate on train set
    """

    # # TODO need to set these equal to the MNIST training batches
    # batch_x = tf.placeholder(tf.float32, [None, SIZE_INPUT], name='Batch x')
    # y_ = tf.placeholder(tf.float32, shape=[None, 10], name ='y')

    # TODO unpack 'weights' into these
    W1 = tf.reshape(tf.slice(weights, [0], [LEN_W1]), [SIZE_INPUT, SIZE_H1])
    b1 = tf.reshape(tf.slice(weights, [LEN_W1], [LEN_B1]), [SIZE_H1])
    W2 = tf.reshape(tf.slice(weights, [LEN_W1+LEN_B1], [LEN_W2]), [SIZE_H1,SIZE_OUTPUT])
    b2 = tf.reshape(tf.slice(weights, [LEN_W1+LEN_B1+LEN_W2], [LEN_B2]), [SIZE_OUTPUT])

    h1 = tf.nn.relu(tf.matmul(batch_x, W1) + b1)
    # h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    yhat = tf.matmul(h1, W2) + b2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=yhat))
    # correct_prediction = tf.equal(tf.argmax(yhat,1), tf.argmax(y_,1))
    return cross_entropy


    # TODO this might have to be modified to correctly reside in the TF graph;
    # we need to replace batch[0] with some large piece of MNIST (the entire train set?) or avg
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # accuracy_eval = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    # return 1 / accuracy_eval # we are returning a value to minimize


# =================================================================================================
# Multi-dimensional quadratic minimization problem

def f_quadratic(x):
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


def init_LSTM():
    # These lines were changed from the original, bc the original didn't compile...
    cell = tf.contrib.rnn.MultiRNNCell( [tf.contrib.rnn.LSTMCell(STATE_SIZE) for _ in xrange(NUM_LAYERS)] )
    cell = tf.contrib.rnn.InputProjectionWrapper(cell, STATE_SIZE)
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
    cell = tf.make_template('cell', cell) # wraps cell function so it does variable sharing
    return cell

def g_sgd(gradients, state, learning_rate = 0.1):
    """ Optimizer: stochastic gradient descent. """
    return -learning_rate*gradients, state


def g_rms(gradients, state, learning_rate = 0.1, decay_rate = 0.99):
    """ Optimizer: RMSProp """
    if state is None:
        state = tf.zeros(DIMS)
    state = decay_rate * state + (1 - decay_rate) * tf.pow(gradients, 2)
    update = -learning_rate * gradients / ( tf.sqrt(state) + 1e-5 )
    return update, state

# cell = init_LSTM() # was a global before

def g_rnn(gradients, state):
    """ Optimizer: Our custom LSTM """
    gradients = tf.expand_dims(gradients, axis=1)
    if state is None:
        state = [ [tf.zeros([DIMS, STATE_SIZE])] * 2 ] * NUM_LAYERS
    cell = init_LSTM() # was a global before
    update, state = cell(gradients, state)
    return tf.squeeze(update, axis=[1]), state # No idea what squeeze does...

# =================================================================================================

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def initialize_x():
    # W_1 = weight_variable([SIZE_INPUT, SIZE_H1])
    # b_1 = bias_variable([SIZE_H1])
    # W_2 = weight_variable([SIZE_H1, SIZE_H2])
    # b_2 = bias_variable([SIZE_H2])
    # W_3 = weight_variable([SIZE_H2, SIZE_OUTPUT])
    # b_3 = bias_variable([SIZE_OUTPUT])
    return tf.random_uniform([DIMS], -1., 1.)


def learn(optimizer, problem):
    """ Takes an optimizer function, and applies it in a loop (unroll all the training steps)
            for number of steps and collects the value of the function f (loss) at each step.
        Args:
            optimizer: fcn with arguments (gradients, state)
        Returns: A list of losses over the training steps
    """
    losses = []
    x = initialize_x()
    state = None
    for _ in xrange(TRAINING_STEPS):
        loss = problem(x)
        losses.append(loss)
        grads, = tf.gradients(loss, x)

        update, state = optimizer(grads, state)
        x += update
    return losses

def assemble_feed_dict():

    BATCH_SIZE = 128
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    return {
        batch_x: batch_xs,
        y_: batch_ys
    }


def optimize_step(loss):
    """ Returns an ADAM update step to our LSTM on a given loss function.
        Because the entire training loop is in the graph we can use Back-Propagation Through Time 
            (BPTT) and a meta-optimizer to minimize this value. Because everything through the
            computation of the loss function is differentiable, TensorFlow can work out the 
            gradients of the LSTM parameters with respect the sum of the losses on f().
        Returns:
            (tf.Operation) ADAM optimization step for the LSTM
    """
    optimizer = tf.train.AdamOptimizer(0.0001) # learning rate
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.) # important because some values are large
    return optimizer.apply_gradients(zip(gradients, v))


def train_LSTM(sess, sum_losses, apply_update):
    """ Train the LSTM. """
    print('Train LSTM...')
    sess.run(tf.global_variables_initializer())
    ave = 0
    for i in xrange(100):
        err, _ = sess.run([sum_losses, apply_update],
            feed_dict= assemble_feed_dict())
        ave += err
        if i % 10 == 0:
            print(ave / 10 if i!=0 else ave)
            ave = 0


# =================================================================================================
# Plotting (these can be refactored into a single function)

def display_LSTM(sess, loss_list, n_times):
    """ Evaluate on the same problem"""
    print('Evaluate LSTM cost tensors...')
    assert len(loss_list) == 3; 'loss_list should have 3 components'
    x = np.arange(TRAINING_STEPS)
    for _ in range(n_times):
        sgd_1, rms_1, rnn_1 = sess.run(loss_list, feed_dict = assemble_feed_dict()) # evaluate loss tensors now that LSTM is trained
        p1, = plt.plot(x, sgd_1, label='SGD')
        p2, = plt.plot(x, rms_1, label='RMS')
        p3, = plt.plot(x, rnn_1, label='RNN')
        plt.legend(handles=[p1, p2, p3])
        plt.title('Losses')
        # plt.show()
        now = time.strftime("%H%M%S")
        plt.savefig('./images/lstm_result_' + now)


def display_base_optimizers(sess, loss_list, n_times):
    """Sanity check of SGD vs RMS, test to make sure learn() is working"""
    print('Sanity check: evaluate basic optimizer tensors...')
    assert len(loss_list) == 2; 'loss_list should have 2 components'
    x = np.arange(TRAINING_STEPS)
    for _ in xrange(n_times):
        sgd_1, rms_1 = sess.run(loss_list, feed_dict = assemble_feed_dict()) # evaluate loss tensors
        p1, = plt.plot(x, sgd_1, label='SGD')
        p2, = plt.plot(x, rms_1, label='RMS')
        plt.legend(handles=[p1,p2])
        plt.title('Losses')
        # plt.show()
        now = time.strftime("%H%M%S")
        plt.savefig('./images/base_result_' + now)



# =================================================================================================


def main():
    print('Initializing...')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print('Assemble computation graph...')
    # TODO change to f_neural
    problem = f_neural
    sgd_losses = learn(g_sgd, f_neural)
    rms_losses = learn(g_rms, f_neural)
    rnn_losses = learn(g_rnn, f_neural)
    sum_losses = tf.reduce_sum(rnn_losses)
    apply_update = optimize_step(sum_losses)
    display_base_optimizers(sess, loss_list=[sgd_losses, rms_losses], n_times=1)
    train_LSTM(sess, sum_losses, apply_update)
    display_LSTM(sess, loss_list=[sgd_losses, rms_losses, rnn_losses], n_times=1)


if __name__ == '__main__':
    main()
