# CS525_Project

	/metalearn
		metalearn.py 	Optimizer LSTM
		network.py 		Optimizee network (feedforward ANN with one hidden layer)
	runner.py 			Runs the program (optimizes the LSTM)

## Usage

To run a test:
	
	$ python runner.py

## Overview

`learn_optimizee` returns a list of costs over the course of training a single optimizee network.

	unrolled_losses <- learn(optimizer, problem, batch)
	sum_losses <- reduce_sum(unrolled_losses)
	apply_update = optimize_step(sum_losses)
	train_LSTM(session, sum_losses, apply_update, batch)
	evaluate_LSTM(sess, unrolled_losses, batch)

## TODO

* Architecture Generalization tests
	* different activation functions: Sigmoid, tanh, relu, leaky relu, elu
	* different # of hidden units
	* different # of hidden layers (hard)
* Advanced generalization
	* test that same LSTM on a different dataset (the smile dataset) to see if it generalizes, and do the reverse as well (see if a LSTM trained on smile problem generalizes to MNIST)
* Advanced architectures
	* convnet on MNIST