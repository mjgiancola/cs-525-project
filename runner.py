"""
    runner.py
    30 April 2017

"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from metalearn.network import PROBLEM_NEURAL, PROBLEM_QUADRATIC, NN_PROBLEMS, PROBLEM_NEURAL_SIGMOID, PROBLEM_NEURAL_TANH

# =================================================================================================
# Globals

# Import MNIST data set
print('Loading training data...')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
prefix = 'smile_data/'
trainingFaces = np.load(prefix + "trainingFaces.npy")
trainingLabels = np.load(prefix + "trainingLabels.npy")
testingFaces = np.load(prefix + "testingFaces.npy")
testingLabels = np.load(prefix + "testingLabels.npy")

# DIMS = 10 # dimensionality of cost function space; equal to number of optimizee params
# scale = tf.random_uniform([DIMS], 0.5, 1.5)

TRAINING_STEPS = 100 # 100 in the paper ?
TRAIN_LSTM_STEPS = 100

# LSTM params
NUM_LAYERS = 2
STATE_SIZE = 20 # n of hidden units

lstm_variables = None

# =================================================================================================


def init_LSTM():
    with tf.variable_scope("LSTM") as vs:
        # These lines were changed from the original, bc the original didn't compile...
        cell = tf.contrib.rnn.MultiRNNCell( [tf.contrib.rnn.LSTMCell(STATE_SIZE) for _ in xrange(NUM_LAYERS)] )
        cell = tf.contrib.rnn.InputProjectionWrapper(cell, STATE_SIZE)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
        cell = tf.make_template('cell', cell) # wraps cell function so it does variable sharing
        # print(vs.name)
        lstm_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
        return cell

def g_sgd(gradients, state, DIMS, learning_rate = 0.1):
    """ Optimizer: stochastic gradient descent. """
    return -learning_rate*gradients, state


def g_rms(gradients, state, DIMS, learning_rate = 0.1, decay_rate = 0.99):
    """ Optimizer: RMSProp """
    if state is None:
        state = tf.zeros(DIMS)
    state = decay_rate * state + (1 - decay_rate) * tf.pow(gradients, 2)
    update = -learning_rate * gradients / ( tf.sqrt(state) + 1e-5 )
    return update, state

cell = init_LSTM()

def g_rnn(gradients, state, DIMS):
    """ Optimizer: Our custom LSTM """
    gradients = tf.expand_dims(gradients, axis=1)
    if state is None:
        state = [ [tf.zeros([DIMS, STATE_SIZE])] * 2 ] * NUM_LAYERS
    # cell = init_LSTM() # was a global before
    update, state = cell(gradients, state)
    return tf.squeeze(update, axis=[1]), state # No idea what squeeze does...

# =================================================================================================

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def initialize_x(DIMS):
    return tf.random_uniform([DIMS], -1., 1.)


def learn_optimizee(optimizer, problem, batch):
    """ Takes an optimizer function, and applies it in a loop (unroll all the training steps)
            for number of steps and collects the value of the function f (loss) at each step.
        Args:
            optimizer: fcn with arguments (gradients, state)
        Returns: A list of losses over the training steps
    """
    losses = []
    x = initialize_x(problem.DIMS)
    state = None
    for _ in xrange(TRAINING_STEPS):
        loss = problem.loss_fcn(x, batch=batch)
        losses.append(loss)
        grads, = tf.gradients(loss, x)
        update, state = optimizer(grads, state, problem.DIMS)
        x += update
    return losses


def shuffleArraysInUnison(x, y):
    assert x.shape[0] == y.shape[0]
    p = np.random.permutation(x.shape[0])
    return x[p, :], y[p]


def get_train_smile(batch_size):
    x, y = shuffleArraysInUnison(trainingFaces, trainingLabels)
    return x[:batch_size, :], y[:batch_size]


def get_test_smile(batch_size):
    x, y = shuffleArraysInUnison(testingFaces, testingLabels)
    return x[:batch_size, :], y[:batch_size]


def assemble_batch_template(batch, fcn):
    batch_x, batch_y = batch[0], batch[1]
    BATCH_SIZE = 128
    batch_xs, batch_ys = fcn(BATCH_SIZE)
    return {
        batch_x: batch_xs,
        batch_y: batch_ys
    }

# Not used for now
def evaluate_accuracy_with_validation_set(sess, accuracy, batch):
    batch_x, batch_y = batch[0], batch[1]
    batch_xs, batch_ys = mnist.validation.images, mnist.validation.labels
    measured_accuracy = sess.run(accuracy, feed_dict = {
        batch_x: batch_xs,
        batch_y: batch_ys })
    return measured_accuracy


def assemble_train_mnist(batch):
    return assemble_batch_template(batch, mnist.train.next_batch)

def assemble_test_mnist(batch):
    return assemble_batch_template(batch, mnist.test.next_batch)

def assemble_train_smile(batch):
    return assemble_batch_template(batch, get_train_smile)

def assemble_test_smile(batch):
    return assemble_batch_template(batch, get_test_smile)


# def assemble_test_mnist(batch):
#     batch_x, batch_y = batch[0], batch[1]
#     BATCH_SIZE = 128
#     batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE)
#     return {
#         batch_x: batch_xs,
#         batch_y: batch_ys
#     }


def optimize_step(loss):
    """ Returns an ADAM update step to our LSTM on a given loss function.
        Because the entire training loop is in the graph we can use Back-Propagation Through Time 
            (BPTT) and a meta-optimizer to minimize this value. Because everything through the
            computation of the loss function is differentiable, TensorFlow can work out the 
            gradients of the LSTM parameters with respect the sum of the losses on f().
        Returns:
            (tf.Operation) ADAM optimization step for the LSTM
    """
    optimizer = tf.train.AdamOptimizer(0.001) # learning rate
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.) # important because some values are large
    return optimizer.apply_gradients(zip(gradients, v))


def train_LSTM(sess, sum_losses, apply_update, batch):
    """ Train the LSTM. """
    print('Train LSTM...')
    # writer = tf.train.SummaryWriter('./tmp/nn_logs')
    # merged = tf.merge_all_summaries()

    sess.run(tf.global_variables_initializer())
    # ave = 0
    for i in xrange(TRAIN_LSTM_STEPS):
        err, _ = sess.run([sum_losses, apply_update],
            feed_dict= assemble_train_mnist(batch))
        print('Step {} Error: \t {}'.format(i, err))
        # ave += err
        # if i % 10 == 0:
        #     print(ave / 10 if i!=0 else ave)
        #     ave = 0


# =================================================================================================
# Plotting (these can be refactored into a single function)

def evaluate_LSTM(sess, loss_list, n_times, batch):
    """ Evaluate on the same problem. """
    print('Evaluate LSTM cost tensors...')
    assert len(loss_list) == 3; 'loss_list should have 3 components'
    x = np.arange(TRAINING_STEPS)
    for _ in range(n_times):
        sgd_1, rms_1, rnn_1 = sess.run(loss_list, feed_dict=assemble_train_mnist(batch))
        p1, = plt.plot(x, np.log10(sgd_1), label='SGD')
        p2, = plt.plot(x, np.log10(rms_1), label='RMS')
        p3, = plt.plot(x, np.log10(rnn_1), label='RNN')
        plt.legend(handles=[p1, p2, p3])
        plt.title('Losses')
        now = time.strftime("%H%M%S")
        plt.savefig('./images/lstm_result_' + now)
        # plt.show()
        plt.clf()


def display_base_optimizers(sess, loss_list, n_times, batch):
    """Sanity check of SGD vs RMS, test to make sure learn() is working"""
    print('Sanity check: evaluate basic optimizer tensors...')
    assert len(loss_list) == 2; 'loss_list should have 2 components'
    x = np.arange(TRAINING_STEPS)
    for _ in xrange(n_times):
        sgd_1, rms_1 = sess.run(loss_list, feed_dict=assemble_train_mnist(batch)) # evaluate loss tensors
        p1, = plt.plot(x, np.log10(sgd_1), label='SGD')
        p2, = plt.plot(x, np.log10(rms_1), label='RMS')
        plt.legend(handles=[p1,p2])
        plt.title('Losses')
        now = time.strftime("%H%M%S")
        plt.savefig('./images/base_result_' + now)
        # plt.show()
        plt.clf()


# =================================================================================================


def main():
    print('Initializing...')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    problem = PROBLEM_NEURAL
    print('Work on problem {}.'.format(problem.name))
    print('Assemble computation graph...')
    batch_x = tf.placeholder(tf.float32, [None, problem.SIZE_INPUT], name='batch_x')
    batch_y = tf.placeholder(tf.float32, shape=[None, 10], name ='batch_y')
    batch = (batch_x, batch_y)
    sgd_losses = learn_optimizee(g_sgd, problem, batch)
    rms_losses = learn_optimizee(g_rms, problem, batch)
    rnn_losses = learn_optimizee(g_rnn, problem, batch)

    sum_losses = tf.reduce_sum(rnn_losses)
    apply_update = optimize_step(sum_losses)
    display_base_optimizers(sess, loss_list=[sgd_losses, rms_losses], n_times=1, batch=batch)

    saver = tf.train.Saver()
    
    if 1:
        train_LSTM(sess, sum_losses, apply_update, batch)
        print("LSTM model finished training. Saving file...")
        save_path = saver.save(sess, "./tmp/100_steps_0001_model.ckpt",
            lstm_variables)
        print("Model saved in file: %s" % save_path)
    else:
        print("Restoring model from memory...")
        saver.restore(sess, "./tmp/100_steps_0001_model.ckpt")
        print("Model restored.")

    evaluate_LSTM(sess, loss_list=[sgd_losses, rms_losses, rnn_losses], n_times=1, batch=batch)


# def big_test():
#     # evaluate
#     print('Initializing...')
#     sess = tf.InteractiveSession()
#     sess.run(tf.global_variables_initializer())

#     saver = tf.train.Saver()
#     print("Restoring model from memory...")
#     saver.restore(sess, "./tmp/model.ckpt")
#     print("Model restored.")

#     for problem in NN_PROBLEMS:
#         print('Test on problem {}'.format(problem.name))
#         batch_x = tf.placeholder(tf.float32, [None, SIZE_INPUT], name='batch_x')
#         batch_y = tf.placeholder(tf.float32, shape=[None, 10], name ='batch_y')
#         batch = (batch_x, batch_y)
#         sgd_losses = learn(g_sgd, problem, batch)
#         rms_losses = learn(g_rms, problem, batch)
#         rnn_losses = learn(g_rnn, problem, batch)
#         sum_losses = tf.reduce_sum(rnn_losses)
#         apply_update = optimize_step(sum_losses)
#         evaluate_LSTM(sess, loss_list=[sgd_losses, rms_losses, rnn_losses], n_times=1, batch=batch)
#         tf.reset_default_graph() # do this every time to prevent side effect issues
    

if __name__ == '__main__':
    main()
    # big_test()