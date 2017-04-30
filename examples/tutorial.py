import tensorflow as tf

DIMS = 10

scale = tf.random_uniform([DIMS], 0.5, 1.5)

def f(x):
  """ Loss function. """
  x = scale * x
  return tf.reduce_sum(x*x)

def g_sgd(gradients, state, learning_rate = 0.1):
  return -learning_rate*gradients, state

def g_rms(gradients, state, learning_rate = 0.1, decay_rate = 0.99):
  if state is None:
    state = tf.zeros(DIMS)
  state = decay_rate * state + (1 - decay_rate) * tf.pow(gradients, 2)
  update = -learning_rate * gradients / ( tf.sqrt(state) + 1e-5 )
  return update, state

TRAINING_STEPS = 20 # 100 in the paper ?

initial_pos = tf.random_uniform([DIMS], -1., 1.)

def learn(optimizer):
  losses = []
  x = initial_pos
  state = None
  for _ in xrange(TRAINING_STEPS):
    loss = f(x)
    losses.append(loss)
    grads, = tf.gradients(loss, x)

    update, state = optimizer(grads, state)
    x += update
  return losses

sgd_losses = learn(g_sgd)
rms_losses = learn(g_rms)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

x = np.arange(TRAINING_STEPS)
# for _ in xrange(3):
#   sgd_1, rms_1 = sess.run([sgd_losses, rms_losses])
#   p1, = plt.plot(x, sgd_1, label='SGD')
#   p2, = plt.plot(x, rms_1, label='RMS')
#   plt.legend(handles=[p1,p2])
#   plt.title('Losses')
#   plt.show()


NUM_LAYERS = 2
STATE_SIZE = 20

# These lines were changed from the original, bc the original didn't compile...
cell = tf.contrib.rnn.MultiRNNCell( [tf.contrib.rnn.LSTMCell(STATE_SIZE) for _ in xrange(NUM_LAYERS)] )
cell = tf.contrib.rnn.InputProjectionWrapper(cell, STATE_SIZE)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
cell = tf.make_template('cell', cell)

def g_rnn(gradients, state):
  gradients = tf.expand_dims(gradients, axis=1)

  if state is None:
    state = [ [tf.zeros([DIMS, STATE_SIZE])] * 2 ] * NUM_LAYERS
  update, state = cell(gradients, state)
  return tf.squeeze(update, axis=[1]), state # No idea what squeeze does...

rnn_losses = learn(g_rnn)
sum_losses = tf.reduce_sum(rnn_losses)

def optimize(loss):
  optimizer = tf.train.AdamOptimizer(0.0001)
  #Honestly not sure what any of the below lines do
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.)
  return optimizer.apply_gradients(zip(gradients, v))

apply_update = optimize(sum_losses)

sess.run(tf.global_variables_initializer())

ave = 0
for i in xrange(3000):
  err, _ = sess.run([sum_losses, apply_update])
  ave += err
  if i % 1000 == 0:
    print(ave / 1000 if i!=0 else ave)

print ave / 1000 # Why divided by 1000?

for _ in range(3):
  sgd_1, rms_1, rnn_1 = sess.run( [sgd_losses, rms_losses, rnn_losses] )
  p1, = plt.plot(x, sgd_1, label='SGD')
  p2, = plt.plot(x, rms_1, label='RMS')
  p3, = plt.plot(x, rnn_1, label='RNN')
  plt.legend(handles=[p1, p2, p3])
  plt.title('Losses')
  plt.show()