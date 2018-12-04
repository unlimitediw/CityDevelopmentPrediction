import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
np.random.seed(1)

def dataSpecificPro(path):
    datafile = pd.read_csv(path)

    Y = datafile['GDP'].values
    X = datafile.drop(labels = ['CityName','GDP'],axis = 1).values
    #X = datafile[['Area','Population']].values

    def stdScl(F):
        F = (F - np.average(F))/np.std(F)
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
    return X,Y

X, Y = dataSpecificPro('./Data/Final229CitiesData.csv')




# Spilt the test set at the start
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

# Without KFold
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train,Y_train,test_size=0.1)

# Loop to find appropriate c value and gamma value
for i in range(1,1000):
    c_rbf = 400000
    c_poly = 500000
    gamma = np.power(1.3,-5.)
    rbf = svm.SVR(C= c_rbf,kernel='rbf',gamma = gamma,tol = 1e-7)
    poly = svm.SVR(C = c_poly,kernel='poly',gamma = 'auto')
    rbf.fit(X_train,Y_train)
    poly.fit(X_train,Y_train)

    def validation(model,xTest,yTest):
        error = np.sum(abs(model.predict(xTest) - yTest)/yTest) / len(xTest)
        k = model.predict(xTest)
        for i in range(len(xTest)):
            print(format(k[i],'.3f'),format(yTest[i],'.3f'),abs(k[i] - yTest[i])/yTest[i])
        return error

    error = validation(rbf,X_test,Y_test)
    print(error)
    break

# After selecting c and checking with random set of dataset, the error rate is about 50% which is good in some degree
# Now I will try the regression MLP
