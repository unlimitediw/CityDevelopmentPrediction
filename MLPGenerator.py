import numpy as np
import pandas as pd
import itertools

import scipy.optimize

from sklearn.model_selection import train_test_split
from scipy.special import expit

np.random.seed(1)


class MLPGenerator:

    def __init__(self, layerShape, mode, X, Y, X_test, Y_test):
        self.layerShape = layerShape
        self.mode = mode
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test

    # data flatten for sklearn
    def flattenParams(self, Thetas):
        flattened_list = [mytheta.flatten() for mytheta in Thetas]
        combined = list(itertools.chain.from_iterable(flattened_list))
        return np.array(combined).reshape((len(combined), 1))

    def reshapeParams(self, flattened_list):
        start, end = 0, 0
        Thetas = []
        for i in range(len(self.layerShape) - 1):
            start = end
            curShape = (self.layerShape[i + 1], self.layerShape[i] + 1)
            end += curShape[0] * curShape[1]
            theta = flattened_list[start:end].reshape(curShape)
            Thetas.append(theta)
        return Thetas

    def flattenX(self, x):
        return np.array(x.flatten()).reshape((len(x) * (self.layerShape[0] + 1), 1))

    def reshapeX(self, x, preSize):
        return np.array(x).reshape((preSize, self.layerShape[0] + 1))

    # weight generation part
    def genRandThetas(self, epsilon=0.12):

        # attention, theta is at left
        Thetas = []
        for i in range(len(self.layerShape) - 1):
            theta = np.random.rand(*(self.layerShape[i + 1], self.layerShape[i] + 1)) * 2 * epsilon - epsilon
            Thetas.append(theta*50)
        return Thetas

    #
    def propagateForward(self, x, Thetas):
        features = x
        z_memo = []
        for i in range(len(Thetas)):
            theta = Thetas[i]
            # first dot product then reshape, reshape is just for safe and avoid the situation of (n,) rather than (n,1)
            z = theta.dot(features).reshape((theta.shape[0], 1))
            # use sigmoid here, later may change to relu or tanh
            a = expit(z)
            z_memo.append((z, a))
            # print(i,z,a)
            if i == len(Thetas) - 1:
                return np.array(z_memo)
            a = np.insert(a, 0, 1)
            features = a

    def computeCost(self, Thetas, x, y, myLambda):
        Thetas = self.reshapeParams(Thetas)
        train_size = len(x) // (self.layerShape[0] + 1)
        x = self.reshapeX(x, train_size)
        total_cost = 0.

        for i in range(train_size):
            hyper = self.propagateForward(x[i].T, Thetas)[-1][1]
            cost = self.costFunction(hyper, y[i],mode = 'perceptron')
            total_cost += cost
        total_cost = float(total_cost) / train_size
        # in MLP TOTAL regular equals square sum of theta
        total_reg = 0.
        for theta in Thetas:
            total_reg += np.sum(theta * theta)
        total_reg *= float(myLambda) / (2 * train_size)
        return total_cost + total_reg

    def costFunction(self, hyper, y, mode='classification'):
        if mode == 'classification':
            return - (y * np.log(hyper)) - (1 - y) * (np.log(1 - hyper))
        elif mode == 'perceptron':
            print(y*(hyper - y))
            return y * (hyper - y)

    def gradientFunction(self, z, mode='sigmoid'):
        if mode == 'sigmoid':
            # expit = 1/(1+e^z)
            # dummy is the activation layer result
            dummy = expit(z)
            return dummy * (1 - dummy)

    # automatic bp
    def backPropagate(self, Thetas, x, y, my_lambda=0.):
        Thetas = self.reshapeParams(Thetas)
        train_size = len(x) // (self.layerShape[0] + 1)
        X = self.reshapeX(x, train_size)
        Deltas = []
        for i in range(len(self.layerShape) - 1):
            delta = np.zeros((self.layerShape[i + 1], self.layerShape[i] + 1))
            Deltas.append(delta)
        for i in range(train_size):
            zSet = []
            aSet = [X[i].reshape((self.layerShape[0] + 1, 1))]
            temp = self.propagateForward(X[i], Thetas)
            # inherit z,a value from propagateForward function
            for j in range(len(self.layerShape) - 1):
                zSet.append(temp[j][0])
                aSet.append(temp[j][1])
            print(zSet,X)
            deltas = [(zSet[-1] - y[i])]
            print(deltas)
            for i in range(1, len(aSet) - 1):
                t = Thetas[len(Thetas) - i].T[1:,:].dot(deltas[i - 1]) * self.gradientFunction(zSet[len(zSet) - i - 1])
                deltas.append(t)
            deltas = deltas[::-1]
            # delta calculation which dots a will be the lost
            for i in range(1,len(aSet) - 1):
                aSet[i] = np.insert(aSet[i],0,1,axis = 0)
            for i in range(len(Deltas)):
                Deltas[i] += deltas[i].dot(aSet[i].T)
        DSet = [Deltas[k] / train_size for k in range(len(Deltas))]
        for i in range(len(DSet)):
            DSet[i][:,1:] += (my_lambda / train_size) * np.square(Thetas[i][:,1:])
        return self.flattenParams(DSet).flatten()

    def trainNN(self, myLambda=0.):
        Thetas = self.flattenParams(self.genRandThetas())
        result = scipy.optimize.fmin_cg(self.computeCost, x0=Thetas, fprime=self.backPropagate, args=(self.flattenX(self.X), self.Y, myLambda),
                                        maxiter=30,
                                        disp=True, full_output=True)
        return self.reshapeParams(result[0])
