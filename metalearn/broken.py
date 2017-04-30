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

# Import MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# LSTM params
NUM_LAYERS = 2
STATE_SIZE = 20 # n of hidden units
BATCH_SIZE = 128
TOTAL_BATCHES = int(mnist.train.num_examples/BATCH_SIZE)

# =================================================================================================
# Activation functions

def relu(x):
  return np.maximum(x, 0)

def relu_prime(x):
  result = np.zeros(x.shape)
  result[np.nonzero(x>=0)] = 1 # To satisfy check_grad, relu' is 1 at 0
  return result

def soft_max(x):
    row_max = np.max(x, axis = 1,keepdims=True)
    max_removed = x - row_max
    e_x = np.exp(max_removed)
    return e_x / e_x.sum(axis = 1, keepdims=True) 

# =================================================================================================
# Network stuff (can move later -- didn't want to delete what nick had in network.py)

def initialize_weights(hidden_units = 30):
  """
  Initializes weight and bias vectors
  """

  w1_abs = 1.0 / np.sqrt(784)
  w2_abs = 1.0 / np.sqrt(hidden_units)

  W1 =  np.random.uniform(-w1_abs,w1_abs,[hidden_units, 784])
  W2 = np.random.uniform(-w2_abs,w2_abs,[10, hidden_units])
  
  b1 = 0.01 * np.ones((hidden_units,1))
  b2 = 0.01 * np.ones((10,1))

  return W1, W2, b1, b2

def feed_forward(batch, W1, W2, b1, b2):
  """
  Runs the given batch through the network defined by the given weights
  and return the intermediate values z1, h1, z2 as well as the final y_hats
  """

  # batch is num_instances (varies) * 784
  # W1 is 784 * num_hidden_units (varies)
  # z1 (and h1) should be num_instances * num_hidden_units

  z1_no_bias = np.dot(batch, W1.T)
  z1 =  z1_no_bias + b1.T
  h1 = relu(z1)

  # h1 is num_instances * num_hidden_units
  # W2 is num_hidden_units * 10
  # z2 (and y_hats) should be num_instances * 10

  z2_no_bias = np.dot(h1, W2.T)
  z2 = z2_no_bias + b2.T
  y_hats = soft_max(z2)

  return z1, h1, z2, y_hats

def backprop(batch, batch_labels, z1, h1, z2, y_hats, W1, W2, b1, b2, alpha=0.):
  """
  Runs backpropagation through the network and returns the gradients for 
  J with respect to W1, W2, b1, and b2.
  """

  y_actuals = batch_labels
  m = batch.shape[0]

  dJdz2 = (y_hats - y_actuals) # num_instances * 10
  dJdh1 = np.dot(dJdz2, W2) # num_instances * 3

  # Equivalently, dJ/dz1 (Hadamard Product)
  g = (dJdh1 * relu_prime(z1)).T # num_instances * 30

  # Compute outer product
  dW1 = 1.0 / m * np.dot(g, batch) + alpha*W1
  dW2 = 1.0 / m * np.dot(dJdz2.T, h1) + alpha*W2

  # Gradient is dJ/dz1 * dz1/db1, which is just 1
  db1 = 1.0 / m * np.sum(np.dot(g.T, np.identity(b1.size)).T, axis=1, keepdims=True)

  # Similarly, gradient is dJ/dz2 * dz2/db2, which is also 1
  db2 = 1.0 / m * np.sum(np.dot(dJdz2, np.identity(b2.size)).T, axis=1, keepdims=True)

  return dW1, dW2, db1, db2

# =================================================================================================

def f(y_hats, y_actuals):
  """ Loss function. Represents the network we are trying to optimize, the "optimizee".
      Could also be thought of as the error landscape, or the "problem" we're solving.
  """
  
  # Defined loss as 1 - accuracy... yeah?
  return 1 - (np.mean(np.argmax(y_hats, axis=1) == np.argmax(y_actuals, axis=1)))


def learn(optimizer):
    """ Takes an optimizer function, and applies it in a loop (unroll all the training steps)
            for number of steps and collects the value of the function f (loss) at each step.
        Args:
            optimizer: fcn with arguments (gradients, state)
        Returns: A list of losses over the training steps
    """
    losses = []
    (W1, W2, b1, b2) = initialize_weights()
    state = None

    for _ in xrange(TRAINING_STEPS):

      # Train on whole randomised dataset looping over batches
      for _ in range(TOTAL_BATCHES):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        
        z1, h1, z2, y_hats = feed_forward(batch_xs, W1, W2, b1, b2)
        loss = f(y_hats, batch_ys)
        losses.append(loss)

        dW1, dW2, db1, db2 = backprop(batch_xs, batch_ys, z1, h1, z2, y_hats, W1, W2, b1, b2)

        print "aaa"
        print dW1

        W1_update, = optimizer(dW1, state)
        W2_update, = optimizer(dW2, state)
        b1_update, = optimizer(db1, state)
        b2_update, = optimizer(db2, state)

        W1 += W1_update
        W2 += W2_update
        b1 += b1_update
        b2 += b2_update

    return losses

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
    for i in xrange(3000):
        err, _ = sess.run([sum_losses, apply_update])
        ave += err
        if i % 1000 == 0:
            print(ave / 1000 if i!=0 else ave)
            if i > 0: ave = 0 # Need to reset so that above actually computes average of most recent 1000


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
    # display_base_optimizers(sess, loss_list=[sgd_losses, rms_losses], n_times=1)
    train_LSTM(sess, sum_losses, apply_update)
    # display_LSTM(sess, loss_list=[sgd_losses, rms_losses, rnn_losses], n_times=1)


if __name__ == '__main__':
    main()
