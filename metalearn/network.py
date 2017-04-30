"""
    network.py
    Nicholas S. Bradford
    30 April 2017

    Train a 2-layer neural network to classify images of hand-written digits from MNIST dataset.
    Implement gradient descent to minimize the cross-entropy loss function.
    Because there are 10 different outputs, there are 10 weights vectors;
        thus, the weight matrix is 784 x 10.

"""

import numpy as np
from sklearn.metrics import accuracy_score


def load_data():
    """ Load data.
        In ipython, use "run -i homework2_template.py" to avoid re-loading of data.
        Each row is an example, each column is a feature.
        Returns:
            train_data      (nd.array): (55000, 784)
            train_labels    (nd.array): (55000, 10)
            test_data       (nd.array): (10000, 784)
            test_labels     (nd.array): (10000, 10)
    """
    prefix = 'mnist_data/'
    train_data = np.load(prefix + "mnist_train_images.npy")
    train_labels = np.load(prefix + "mnist_train_labels.npy")
    test_data = np.load(prefix + "mnist_test_images.npy")
    test_labels = np.load(prefix + "mnist_test_labels.npy")
    assert train_data.shape == (55000, 784) and train_labels.shape == (55000, 10)
    assert test_data.shape == (10000, 784) and test_labels.shape == (10000, 10), str(train_data.shape) + str(test_labels.shape)
    return train_data, train_labels, test_data, test_labels


def J(w, x, y, alpha=0):
    """ Computes cross-entropy loss function.
        J(w1, ..., w10) = -1/m SUM(j=1 to m) { SUM(k=1 to 10) { y } }
        Args:
            w       (np.array): 784 x 10
            x    (np.array): m x 784
            y  (np.array): m x 10
        Returns:
            J (float): cost of w given x and y
    """
    m = x.shape[0]
    assert y.shape[1] == 10, str(x.shape)
    xdotw = x.dot(w)
    bottom_vec = np.exp(xdotw)
    bottom = np.sum(bottom_vec, axis=1, keepdims=True) # sum of each row = sum for each dimension
    top = np.exp(xdotw)
    yhat = np.log(np.divide(top, bottom))
    cost = np.sum(np.multiply(y, yhat))
    return (-1.0 / m) * cost


def gradJ(w, x, y, alpha=0.0):
    """ Compute gradient of cross-entropy loss function. 
        For one training example: dJ/dw = (yhat - yi)x = SUM(1 to m) { yhat_i^(j) - y_i^(j)}
        Args:
            w    (np.array): 784 x 10
            x    (np.array): m x 784
            y    (np.array): m x 10
        Returns:
            grad (np.array): 784 x 10, gradients for each weight in w
    """
    output_dim = y.shape[1]
    m = float(x.shape[0])
    xdotw = x.dot(w)
    bottom_vec = np.exp(xdotw)
    bottom = np.sum(bottom_vec, axis=1) # sum of each row = sum for each dimension
    yhat = np.divide(np.exp(xdotw), bottom[:, None])
    answer = (yhat - y).T.dot(x).T
    assert answer.shape == (784, 10), str(answer.shape)
    return answer / m
    

def gradient_descent(train_data, train_labels, alpha=0.0):
    """ Normally we use Xavier initialization, where weights are randomly initialized to 
            a normal distribution with mean 0 and Var(W) = 1/N_input.
        In this case for SoftMax, we can start with all 0s for initialization.
        Gradient descent is then applied until gain is < epsilon.
        learning_rate =  epsilon 
        threshold = delta
    """
    print('Train 2-layer ANN with regularization alpha: ', alpha)
    w  = np.zeros((train_data.shape[1], train_labels.shape[1]))
    learning_rate = 0.5
    prevJ = J(w, train_data, train_labels, alpha)
    print ('Initial Cost:', prevJ)
    n_iterations = 300
    for i in range(n_iterations):
        update = learning_rate * gradJ(w, train_data, train_labels, alpha)
        w = w - update
        newJ = J(w, train_data, train_labels, alpha)
        diff = prevJ - newJ
        prevJ = newJ
        print('\t{} \tCost: {} \t Diff: {}'.format(i+1, newJ, diff))
    return w


def main():
    train_data, train_labels, test_data, test_labels = load_data()
    w = gradient_descent(train_data, train_labels)
    assert w.shape == (784, 10)
    predictions = test_data.dot(w)
    real_labels = test_labels.argmax(axis=1)
    predict_labels = predictions.argmax(axis=1)
    accuracy = accuracy_score(y_true=real_labels, y_pred=predict_labels)
    loss = J(w, test_data, test_labels)
    print()
    print('Test Loss:     {}'.format(loss))
    print('Test Accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()