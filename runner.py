"""
    runner.py
    Nicholas S. Bradford
    30 April 2017

"""

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt


# =================================================================================================
# Globals

DIMS = 10 # dimensionality of cost function space; equal to number of optimizee params
scale = tf.random_uniform([DIMS], 0.5, 1.5)
TRAINING_STEPS = 20 # 100 in the paper ?

# LSTM params
NUM_LAYERS = 2
STATE_SIZE = 20 # n of hidden units


def init_LSTM():
    # These lines were changed from the original, bc the original didn't compile...
    cell = tf.contrib.rnn.MultiRNNCell( [tf.contrib.rnn.LSTMCell(STATE_SIZE) for _ in xrange(NUM_LAYERS)] )
    cell = tf.contrib.rnn.InputProjectionWrapper(cell, STATE_SIZE)
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
    cell = tf.make_template('cell', cell)
    return cell

cell = init_LSTM()

# =================================================================================================

def f(x):
    """ Loss function. Represents the network we are trying to optimize, the "optimizee".
        Could also be thought of as the error landscape, or the "problem" we're solving.
    """
    x = scale * x
    return tf.reduce_sum(x*x)


def learn(optimizer):
    """ Takes an optimizer function, and applies it in a loop (unroll all the training steps)
            for number of steps and collects the value of the function f (loss) at each step.
        Args:
            optimizer: fcn with arguments (gradients, state)
        Returns: A list of losses over the training steps
    """
    losses = []
    initial_pos = tf.random_uniform([DIMS], -1., 1.)
    x = initial_pos
    state = None
    for _ in xrange(TRAINING_STEPS):
        loss = f(x)
        losses.append(loss)
        grads, = tf.gradients(loss, x)

        update, state = optimizer(grads, state)
        x += update
    return losses

# =================================================================================================


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


def g_rnn(gradients, state):
    """ Optimizer: Our custom LSTM """
    gradients = tf.expand_dims(gradients, axis=1)
    if state is None:
        state = [ [tf.zeros([DIMS, STATE_SIZE])] * 2 ] * NUM_LAYERS
    update, state = cell(gradients, state)
    return tf.squeeze(update, axis=[1]), state # No idea what squeeze does...

# =================================================================================================


def optimize_step(loss):
    """ Returns an ADAM update step to our LSTM on a given loss function.
        Because the entire training loop is in the graph we can use Back-Propagation Through Time 
            (BPTT) and a meta-optimizer to minimize this value! And this is the main point:
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
    for i in xrange(3000):
        err, _ = sess.run([sum_losses, apply_update])
        ave += err
        if i % 1000 == 0:
            print(ave / 1000 if i!=0 else ave)
    print ave / 1000 # Why divided by 1000?


# =================================================================================================
# Plotting (these can be refactored into a single function)

def display_LSTM(sess, loss_list, n_times):
    """ Evaluate on the same problem"""
    print('Evaluate LSTM cost tensors...')
    assert len(loss_list) == 3; 'loss_list should have 3 components'
    x = np.arange(TRAINING_STEPS)
    for _ in range(n_times):
        sgd_1, rms_1, rnn_1 = sess.run(loss_list) # evaluate loss tensors now that LSTM is trained
        p1, = plt.plot(x, sgd_1, label='SGD')
        p2, = plt.plot(x, rms_1, label='RMS')
        p3, = plt.plot(x, rnn_1, label='RNN')
        plt.legend(handles=[p1, p2, p3])
        plt.title('Losses')
        plt.show()


def display_base_optimizers(sess, loss_list, n_times):
    """Sanity check of SGD vs RMS, test to make sure learn() is working"""
    print('Sanity check: evaluate basic optimizer tensors...')
    assert len(loss_list) == 2; 'loss_list should have 2 components'
    x = np.arange(TRAINING_STEPS)
    for _ in xrange(n_times):
        sgd_1, rms_1 = sess.run(loss_list) # evaluate loss tensors
        p1, = plt.plot(x, sgd_1, label='SGD')
        p2, = plt.plot(x, rms_1, label='RMS')
        plt.legend(handles=[p1,p2])
        plt.title('Losses')
        plt.show()

# =================================================================================================


def main():
    print('Initializing...')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print('Assemble computation graph...')
    sgd_losses = learn(g_sgd)
    rms_losses = learn(g_rms)
    rnn_losses = learn(g_rnn)
    sum_losses = tf.reduce_sum(rnn_losses)
    apply_update = optimize_step(sum_losses)
    display_base_optimizers(sess, loss_list=[sgd_losses, rms_losses], n_times=1)
    train_LSTM(sess, sum_losses, apply_update)
    display_LSTM(sess, loss_list=[sgd_losses, rms_losses, rnn_losses], n_times=1)


if __name__ == '__main__':
    main()
