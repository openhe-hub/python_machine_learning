import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = 'ex1data2.txt'


def output(data):
    print(data.head(data.size))


def read():
    data = pd.read_csv(path, header=None, names=['size', 'bedrooms', 'price'])
    print('origin')
    output(data)
    return data


def mean_normalize(data):
    data = (data - data.mean()) / data.std()
    print('after mean normalization')
    output(data)
    return data


def h(x, theta):
    return x * theta.T


def compute_cost(x, y, theta, m):
    cost = np.power((h(x, theta) - y), 2)
    return np.sum(cost) / (2 * m)


def gradient_descent(x, y, theta, alpha, iteration, m, n):
    temp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(iteration + 1)
    cost[0] = compute_cost(x, y, theta,m)
    print(f'origin cost={cost[0]}')

    for i in range(iteration):
        for j in range(n):
            temp[0, j] = theta[0, j] - ((alpha / m) * np.sum(np.multiply((x * theta.T - y), x[:, j])))
        theta = temp
        cost[i + 1] = compute_cost(x, y, theta, m)
        print(f'i={i},cost={cost[i + 1]},theta={theta}')
    return theta, cost


if __name__ == '__main__':
    data = read()
    data = mean_normalize(data)
    data.insert(0, 'one', 1)  # set x0=1

    # subtract data
    n = data.shape[1]
    x = data.iloc[:, 0:n - 1]
    y = data.iloc[:, n - 1:n]
    n -= 1
    m = data.shape[0]
    print(f"m={m},n={n}")

    # convert to matrix
    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.zeros(n))

    # set origin value
    iteration = 1000
    alpha = 0.01

    # gradient_descent
    theta, cost = gradient_descent(x, y, theta, alpha, iteration, m, n)
    print(theta)
