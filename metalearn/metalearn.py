"""
	metalearn.py
	Nicholas S. Bradford
	30 April 2017

"""

import tensorflow as tf


sum_losses is differentiable, gradients flow through the graph we’ve defined just fine! 
TensorFlow is able to work out the gradients of the parameters in our LSTM with respect 
to this sum of losses.  