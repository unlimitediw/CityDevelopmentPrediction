import numpy as np
import pandas as pd
import MLPGenerator
import itertools

# Minimize a function using a nonlinear conjugate gradient algorithm
import scipy.optimize

from sklearn.model_selection import train_test_split
from scipy.special import expit  # Vectorized sigmoid function

np.random.seed(1)

'''
This is similar to the advice for starting with Random Forest and Stochastic Gradient Boosting on a predictive modeling 
problem with tabular data to quickly get an idea of an upper-bound on model skill prior to testing other methods.
'''


def dataSpecificPro(path):
    datafile = pd.read_csv(path)

    Y = datafile['GDP'].values
    X = datafile.drop(labels=['CityName', 'GDP'], axis=1).values

    # X = datafile[['Area','Population']].values

    def stdScl(F):
        F = (F - np.average(F)) / np.std(F)
        return F

    X = X.T
    X[0] = stdScl(X[0])
    X[1] = stdScl(X[1])
    X[3] = stdScl(X[3])
    X[4] = stdScl(X[4])
    X[5] = stdScl(X[5])
    X[6] = stdScl(X[6])
    X[7] /= 180
    X[8] /= 180
    X = X.T
    return X, Y


X, Y = dataSpecificPro('./Data/Final229CitiesData.csv')
X = np.insert(X, 0, 1, axis=1)
Y /= 10000
'''
Data pre processing
'''
# Spilt the test set at the start
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Without KFold
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.1)

MLPG = MLPGenerator.MLPGenerator([9,6,6,6,1],None,X_train,Y_train,X_validation,Y_validation)

def predictNN(x, Thetas):
    output = MLPG.propagateForward(x, Thetas)[-1][0]
    return output


def computeAccuracy(Thetas, x, y):
    total = x.shape[0]
    res = 0.

    for i in range(total):
        #print(len(x[i]),Thetas)
        hyper = predictNN(x[i], Thetas)
        score = abs(hyper - y[i]) / y[i]
        print(score,hyper,y[i])
        res += score
    res /= total
    return "%0.1f%%" % res

learn = MLPG.trainNN()
print(computeAccuracy(learn, X_validation, Y_validation))