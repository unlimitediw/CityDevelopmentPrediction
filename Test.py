import numpy as np
import pandas as pd
import MLPGenerator
import itertools
np.random.seed(3)

# Minimize a function using a nonlinear conjugate gradient algorithm
import scipy.optimize

from sklearn.model_selection import train_test_split
from scipy.special import expit  # Vectorized sigmoid function

np.random.seed(1)

'''
This is similar to the advice for starting with Random Forest and Stochastic Gradient Boosting on a predictive modeling 
problem with tabular data to quickly get an idea of an upper-bound on model skill prior to testing other methods.
'''

'''
part of MLP
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
indices = np.random.choice(X.shape[0],X.shape[0],replace=False)
X = X[indices]
Y = Y[indices]
'''
Data pre processing
'''
# Spilt the test set at the start
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.001)
mean = np.mean(Y_train)
Y_train /= mean


'''
np.random.seed(11)
for i in range(2):
    X_train = np.asarray(list(X_train) + list(X_train))
    Y_train = np.asarray(list(Y_train) + list(Y_train))
indices = np.random.choice(X_train.shape[0],X_train.shape[0],replace=True)
X_train = X_train[indices]
Y_train = Y_train[indices]
np.random.seed(15)
indices = np.random.choice(X_train.shape[0],X_train.shape[0],replace=True)
X_train = X_train[indices]
Y_train = Y_train[indices]
np.random.seed(17)
indices = np.random.choice(X_train.shape[0],X_train.shape[0],replace=True)
X_train = X_train[indices]
Y_train = Y_train[indices]
'''

# Without KFold
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.8)
a = np.mean(Y_train)
#MLPG = MLPGenerator.MLPGenerator([9,6,1],None,X_train,Y_train,X_validation,Y_validation,np.mean(Y_train))

def predictNN(x, Thetas):
    output = MLPG.propagateForward(x, Thetas)[-1][0]
    return output


def computeAccuracy (Thetas, x, y):
    total = x.shape[0]
    res = 0.
    error = 0
    for i in range(total):
        #print(len(x[i]),Thetas)
        hyper = max(15000,predictNN(x[i], Thetas) * mean)
        score = abs(hyper - y[i]*mean) / (y[i]*mean)
        print(hyper,y[i]*mean)
        error += score
    error /= total
    return error

import KFoldValidation as KF
dataset = KF.KFold(X_train,Y_train)

if "mlp" == "mlp":
    averageError = 0
    averageTrainError = 0
    for i in range(1, 11):
        X_train, Y_train, X_validation, Y_validation = dataset.spilt(i)
        MLPG = MLPGenerator.MLPGenerator([9,15,25,12,6,1],None,X_train,Y_train,None,None,np.mean(Y_train))
        learn = MLPG.trainNN()
        error = computeAccuracy(learn,X_validation,Y_validation)
        trainError = computeAccuracy(learn,X_train,Y_train)
        averageError += error
        averageTrainError += trainError
        print(error,trainError)
        break
    print("validationError:",averageError)
    print("TrainError",averageTrainError)

    learn = MLPG.trainNN()
    #print(computeAccuracy(learn, X_test, Y_test))

    #print("validation")
    #print(computeAccuracy(learn, X_validation, Y_validation*mean))
    #print(computeAccuracy(learn, X_validation, Y_validation))

    # 大数惩罚
    # bagging
    # I face so many problem, the most important one is the label scale and model change attention to a and z
    # cost function can not be square here
    # change the cost function from (y-hyper) -> +-(y-hyper) -> **0.5/**2 -> /y -> /max(hyper,y)


'''
part of data plot
'''
if 1 == 2:
    # CityName,Area,GreenAreaPerPers,Polycentricity,PopulationInCore,#Gov,#GovInCore,Population,Latitude,Longitude,GDP
    LosAngeles = [83682.18,5.09,1,100.0,169,169,3884307,34.0194,-118.4108,891793.72]
    Eindhoven = [1199.68,827.0,2,44.85,19,2,209170,51.441642,5.469722,31087.24]
    X = ['Area', 'GreenArea','Polycentricity','PopulationInCore','#Gov','#GovInCore','Population']
    Y1 = [83682.18,5.09,1,100.0,169,169,3884307]
    Y2 = [1199.68,827.0,2,44.85,19,2,209170]
    GDP1 = 891793.72
    GDP2 = 31087.24
    index = np.arange(7)
    for i in range(len(Y1)):
        Y1[i] = float('%.3f' % np.log2(Y1[i]))
        Y2[i] = float('%.3f' % np.log2(Y2[i]))
    Y1 = tuple(Y1)
    Y2 = tuple(Y2)

    ### 柱状图画法 记得收集
    print(Y2)
    import matplotlib.pyplot as plt
    print(Y1)
    plt.figure(figsize=(9,5))
    bar_width = 0.25
    opacity = 0.8
    #fig, ax = plt.subplots()
    plt.ylabel("Log2 Features Value Comparison")
    plt.title("LosAngeles vs Eindhoven")
    plt.bar(index,Y1,bar_width,alpha = opacity, color = 'C1',label = 'LosAngeles')
    plt.bar(index + bar_width,Y2,bar_width,alpha = opacity, color ='C2',label = 'Eindhoven')
    plt.xticks(index+bar_width,('Area', 'GreenArea','Polycentricity','PopulationInCore','#Gov','#GovInCore','Population'))
    plt.tight_layout()
    plt.legend()
    plt.show()

    ### 饼图画法
    plt.pie([GDP1,GDP2],colors=['C1','C2'],explode=[0,0.1], labels=["LosAngeles","Eindhoven"],shadow=True,autopct="%1.1f%%",pctdistance=0.8)
    plt.title("GDP Comparison of LosAngeles and Eindhoven")
    plt.show()

def stdScl(F):
    F = (F - np.average(F)) / np.std(F)
    return F


if "dist" == "distl":
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    sns.set(color_codes=True)
    datapath = "/Users/unlimitediw/PycharmProjects/CitiesPrediction/Data/PopulationLabelC.csv"
    datapath2 = "/Users/unlimitediw/PycharmProjects/CitiesPrediction/Data/Final229CitiesData.csv"
    data = pd.read_csv(datapath).values
    data2 = pd.read_csv(datapath2).values.T[1]
    x = np.array(data2,dtype=float)
    x //= 1000
    otherMemo = []
    for cur_x in x:
        otherMemo.append(cur_x)
    print(otherMemo)
    sns.distplot(otherMemo,bins=40)
    plt.title("City Area Distribution")
    plt.xlabel("City Area Level (*1000km^2)")
    plt.ylabel("Distribution")
    plt.show()
