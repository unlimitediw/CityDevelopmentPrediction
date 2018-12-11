import numpy as np
import pandas as pd
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

'''
Data pre processing
'''
# Spilt the test set at the start
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Without KFold
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.1)
m = len(X_train)

# Method of cg or handwork
method = 'cg'

# In this place Layer setting will combine intuition and experiment

'''
MLP layer setting
'''
input_layer_size = 9  # Also attention. In perceptron, we need to insert 1
# I want to try 3 hidden layer in this experiment and see what will be going on
hidden_layer_1_size = 9
hidden_layer_2_size = 6
hidden_layer_3_size = 3
output_layer_size = 1


# 针对fmin_cg的flatten和reshape
def flattenParams(Thetas):
    flattened_list = [mytheta.flatten() for mytheta in Thetas]
    combined = list(itertools.chain.from_iterable(flattened_list))
    return np.array(combined).reshape((len(combined), 1))


def reshapeParamsOld(flattened_list):
    size_1 = hidden_layer_1_size * (input_layer_size + 1)
    size_2 = hidden_layer_2_size * (hidden_layer_1_size + 1)
    size_3 = hidden_layer_3_size * (hidden_layer_2_size + 1)
    theta1 = flattened_list[:size_1].reshape(
        (hidden_layer_1_size, input_layer_size + 1))
    theta2 = flattened_list[size_1: size_1 + size_2].reshape(
        (hidden_layer_2_size, hidden_layer_1_size + 1))
    theta3 = flattened_list[size_1 + size_2: size_1 + size_2 + size_3].reshape(
        (hidden_layer_3_size, hidden_layer_2_size + 1))
    theta4 = flattened_list[size_1 + size_2 + size_3:].reshape(
        (output_layer_size, hidden_layer_3_size + 1))

    return [theta1, theta2, theta3, theta4]

layerShape = [9,9,6,3,1]
def reshapeParams(flattened_list):
    start, end = 0, 0
    Thetas = []
    for i in range(len(layerShape) - 1):
        start = end
        curShape = (layerShape[i+1],layerShape[i] + 1)
        end += curShape[0] * curShape[1]
        cur = flattened_list[start:end].reshape(curShape)
        Thetas.append(cur)
    return Thetas

def flattenX(x):
    train_size = len(x)
    return np.array(x.flatten()).reshape((train_size*(input_layer_size+1),1))

def reshapeX(x,preSize):
    return np.array(x).reshape((preSize,input_layer_size+1))



def genRandThetas():
    epsilon = 0.12
    # attention, theta is at left
    theta1_shape = (hidden_layer_1_size, input_layer_size + 1)
    theta2_shape = (hidden_layer_2_size, hidden_layer_1_size + 1)
    theta3_shape = (hidden_layer_3_size, hidden_layer_2_size + 1)
    theta4_shape = (output_layer_size, hidden_layer_3_size + 1)

    def generateTheta(theta_shape):
        # np.random.rand: Create an array of the given shape and populate it with random samples from a uniform distribution over [0,1)
        # The * unpacks a tuple into multiple input arguments
        return np.random.rand(*theta_shape) * 2 * epsilon - epsilon

    return generateTheta(theta1_shape), generateTheta(theta2_shape), generateTheta(theta3_shape), generateTheta(theta4_shape)


def propagateForward(x, Thetas):
    # Thetas = [theta1, theta2, theta3, theta4]
    features = x
    z_memo = []
    for i in range(len(Thetas)):
        theta = Thetas[i]
        # first dot product then reshape, reshape is just for safe and avoid the situation of (n,) rather than (n,1)
        z = theta.dot(features).reshape((theta.shape[0], 1))
        # use sigmoid here, later may change to relu or tanh
        a = expit(z)
        z_memo.append((z, a))
        #print(i,z,a)
        if i == len(Thetas) - 1:
            return np.array(z_memo)
        a = np.insert(a, 0, 1)
        features = a


def computeCost(Thetas, x, y, myLambda):
    if method == 'cg':
        Thetas = reshapeParams(Thetas)
        x = reshapeX(x,m)
    total_cost = 0.
    train_size = m
    for i in range(train_size):
        hyper = propagateForward(x[i].T, Thetas)[-1][1]
        cost = -y[i] * (hyper - y[i])
        total_cost += cost
    total_cost = float(total_cost)/ train_size
    # in MLP TOTAL regular equals square sum of theta
    total_reg = 0.
    for theta in Thetas:
        total_reg += np.sum(theta * theta)
    total_reg *= float(myLambda) / (2 * train_size)
    return total_cost + total_reg


def specialGradient(z, type='sigmoid'):
    if type == 'sigmoid':
        # expit = 1/(1+e^z)
        # dummy is the activation layer
        dummy = expit(z)
        return dummy * (1 - dummy)


def backPropagateOld(Thetas, x, y, myLambda):
    if method == 'cg':
        #print(1)
        Thetas = reshapeParams(Thetas)
        x = reshapeX(x,m) # m is the preset value of len(X_train)
    train_size = len(x)
    # The capital-delta matrix is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative
    # Initialize Delta with all 0
    Delta1 = np.zeros((hidden_layer_1_size, input_layer_size + 1))
    Delta2 = np.zeros((hidden_layer_2_size, hidden_layer_1_size + 1))
    Delta3 = np.zeros((hidden_layer_3_size, hidden_layer_2_size + 1))
    Delta4 = np.zeros((output_layer_size, hidden_layer_3_size + 1))
    for i in range(train_size):
        # There are one input layer, 3 hidden layer, 1 output layer. so a1-5
        a1 = x[i].reshape((input_layer_size + 1, 1))
        # generate a2-4
        temp = propagateForward(x[i], Thetas)
        z2 = temp[0][0]
        a2 = temp[0][1]
        z3 = temp[1][0]
        a3 = temp[1][1]
        z4 = temp[2][0]
        a4 = temp[2][1]
        z5 = temp[3][0]
        a5 = temp[3][1]
        # backward
        # output layer
        delta5 = (z5 - y[i]) / 50
        print(z5,y[i],delta5)
        # 3rd hidden layer (remember to remove the adding 1 position)
        # theta4.dot(hyper - y) * sigmoidGradient(z4)
        delta4 = Thetas[3].T[1:, :].dot(delta5) * specialGradient(z4)
        a4 = np.insert(a4, 0, 1, axis=0)
        # 2nd hidden layer
        delta3 = Thetas[2].T[1:, :].dot(delta4) * specialGradient(z3)
        a3 = np.insert(a3, 0, 1, axis=0)
        # 1st hidden layer
        delta2 = Thetas[1].T[1:, :].dot(delta3) * specialGradient(z2)
        a2 = np.insert(a2, 0, 1, axis=0)
        # input layer do not need the delta
        # Update the Deltas
        Delta1 += delta2.dot(a1.T)
        Delta2 += delta3.dot(a2.T)
        Delta3 += delta4.dot(a3.T)
        Delta4 += delta5.dot(a4.T)
    D1 = Delta1 / train_size
    D2 = Delta2 / train_size
    D3 = Delta3 / train_size
    D4 = Delta4 / train_size
    # Add regulation part
    D1[:, 1:] += (myLambda / train_size) * np.square(Thetas[0][:, 1:])
    D2[:, 1:] += (myLambda / train_size) * np.square(Thetas[1][:, 1:])
    D3[:, 1:] += (myLambda / train_size) * np.square(Thetas[2][:, 1:])
    D4[:, 1:] += (myLambda / train_size) * np.square(Thetas[3][:, 1:])
    # (9, 10) (6, 10) (3,7) (1,4) 不能被装在一个ndarray里，可能需要flatten
    # 由于 scipy.fmin_cg 只支持np.ndarray格式导入导出，所以需要 flatten&reshape
    # 由于args的限制所以这里将手动调节模式
    #print(D1,D2,D3,D4)
    if method == 'cg':
        return flattenParams([D1, D2, D3, D4]).flatten()
    else:
        return np.asarray([D1, D2, D3, D4])


def backPropagate(Thetas, x, y, my_lambda=0.):
    Thetas = reshapeParams(Thetas)
    train_size = len(x) // (layerShape[0] + 1)
    x = reshapeX(x, train_size)
    Deltas = []
    for i in range(len(layerShape) - 1):
        delta = np.zeros((layerShape[i + 1], layerShape[i] + 1))
        Deltas.append(delta)
    for i in range(train_size):
        zSet = []
        aSet = [X[i].reshape((layerShape[0] + 1, 1))]
        temp = propagateForward(x[i], Thetas)
        for j in range(len(layerShape) - 1):
            zSet.append(temp[j][0])
            aSet.append(temp[j][1])
        # delta is just a diff, Delta is gradient
        # bp should remove first X0
        # Thetas[1].T[1:,:].dot(delta3) is the theta.dot(pre_error)
        deltas = [aSet[-1] - y[i]]
        for i in range(1,len(aSet) - 1):
            print(Thetas[len(Thetas) - i].T[1:,:].shape,deltas[i - 1].shape,zSet[len(zSet) - i -1].shape)
            t = Thetas[len(Thetas) - i].T[1:,:].dot(deltas[i - 1]) * specialGradient(zSet[len(zSet) - i - 1])
            deltas.append(t)
        deltas = deltas[::-1]
        for i in range(1,len(aSet) - 1):
            aSet[i] = np.insert(aSet[i],0,1,axis = 0)
        for i in range(len(Deltas)):
            Deltas[i] += deltas[i].dot(aSet[i].T)
    DSet = [Deltas[k] / train_size for k in range(len(Deltas))]
    for i in range(len(DSet)):
        print(Thetas[i].shape)
        DSet[i][:,1:] += (my_lambda / train_size) * np.square(Thetas[i][:,1:])
    return flattenParams(DSet).flatten()



def trainNN(x, y, myLambda=0.):
    Thetas = flattenParams(genRandThetas())
    result = scipy.optimize.fmin_cg(computeCost, x0=Thetas, fprime=backPropagate, args=(flattenX(x), y, myLambda), maxiter=10,
                                    disp=True, full_output=True)
    return reshapeParams(result[0])


def predictNN(x, Thetas):
    output = propagateForward(x, Thetas)[-1][0]
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


learn = trainNN(X_train, Y_train)
print("!!!")
print(computeAccuracy(learn, X_validation, Y_validation))
# 现在的问题是 1:cost function design 2:propagate forward output layer design, no expit