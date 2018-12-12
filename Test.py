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

if 1 == 2:
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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
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
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.1)
    a = np.mean(Y_train)
    MLPG = MLPGenerator.MLPGenerator([9,6,1],None,X_train,Y_train,X_validation,Y_validation,a)

    def predictNN(x, Thetas):
        output = MLPG.propagateForward(x, Thetas)[-1][0]
        return output


    def computeAccuracy (Thetas, x, y):
        total = x.shape[0]
        res = 0.

        for i in range(total):
            #print(len(x[i]),Thetas)
            hyper = max(5000,predictNN(x[i], Thetas) * mean)
            score = abs(hyper - y[i]) / max(y[i],hyper)
            print("!!!!!!!!!!!!!")
            print(score,hyper,y[i],)
            #print("----------")
            #print(x[i])
            #print("----------")
            #print(Thetas)
            print("!!!!!!!!!!")

            res += score
        res /= total
        return "%0.1f%%" % res


    learn = MLPG.trainNN()
    print(computeAccuracy(learn, X_test, Y_test))

    print("validation")
    print(computeAccuracy(learn, X_validation, Y_validation*mean))
    #print(computeAccuracy(learn, X_validation, Y_validation))

    # 大数惩罚
    # bagging
    # I face so many problem, the most important one is the label scale and model change attention to a and z
    # cost function can not be square here
    # change the cost function from (y-hyper) -> +-(y-hyper) -> **0.5/**2 -> /y -> /max(hyper,y)


'''
part of data plot
'''
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