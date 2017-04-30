# CS525_Project

	/metalearn
		metalearn.py 	Optimizer LSTM
		network.py 		Optimizee network (feedforward ANN with one hidden layer)
	runner.py 			Runs the program (optimizes the LSTM)

## Usage

To run a test:
	
	$ python runner.py

## TODO

* integrate our old MNIST homework with the simple code that Mike has to replicate those results
* test that same LSTM on a different dataset (the smile dataset) to see if it generalizes, and do the reverse as well (see if a LSTM trained on smile problem generalizes to MNIST)
* test that LSTM on same problem but slightly different optimizee network (i.e. test out a bunch of different activation functions: Sigmoid, tanh, relu, leaky relu, elu) and see if it generalizes