import numpy as np


def regression(x, y):
    w = np.linalg.lstsq(x, y, rcond=None)[0]
    return w


def cost_function(x, y, theta):
    return np.sum((np.dot(x, theta) - y) ** 2) / 2 * len(y)


def ridge_cost_function(x, y, theta, alpha):
    return np.sum(((np.dot(x, theta) - y) ** 2 + (alpha * theta))) / len(y)


def gradient_decent(x, y, learning_rate, theta, epochs):
    cost_history = []

    for i in range(epochs):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = np.dot(x.T, loss) / len(y)
        theta = theta - learning_rate * gradient
        cost = cost_function(x, y, theta)
        cost_history.append(cost)

    return theta, cost_history


def ridge_regression(x, y, learning_rate, theta, alpha, epochs):
    # cost_history = []

    for i in range(epochs):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = 2 * (np.dot(x.T, loss) + alpha * theta) / len(y)
        theta = theta - learning_rate * gradient
        # cost = ridge_cost_function(x, y, theta, alpha)
        # cost_history.append(cost)

    return theta


def lasso_regression(x, y, learning_rate, theta, alpha, epochs):
    for i in range(epochs):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = (2 * np.dot(x.T, loss) + alpha * np.abs(theta) / theta) / len(y)
        theta = theta - learning_rate * gradient

    return theta


def fit(x, y, learning_rate, epochs):
    theta = np.zeros(x.shape[1])
    theta, cost = gradient_decent(x, y, learning_rate, theta, epochs)

    return theta, cost


def ridge_fit(x, y, learning_rate, alpha, epochs):
    theta = np.zeros(x.shape[1])
    theta = ridge_regression(x, y, learning_rate, theta, alpha, epochs)

    return theta


def lasso_fit(x, y, learning_rate, alpha, epochs):
    theta = np.ones(x.shape[1])
    theta = lasso_regression(x, y, learning_rate, theta, alpha, epochs)

    return theta
