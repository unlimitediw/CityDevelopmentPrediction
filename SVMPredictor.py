import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
np.random.seed(6)
kudo = []
KUFO =  np.asarray([[790.85,37.13,1,51.88,11,2,193127,35.6623,138.5682]])
def dataSpecificPro(path):
    datafile = pd.read_csv(path)

    Y = datafile['GDP'].values
    X = datafile.drop(labels = ['CityName','GDP'],axis = 1).values
    #X = datafile[['Area','Population']].values

    def stdScl(F):
        F = (F - np.average(F))/np.std(F)
        return F

    def FS(F):
        F = (F - min(F))/(max(F) - min(F))
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
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)
def validation(model,xTest,yTest):
    hypo = model.predict(xTest)
    error = 0
    for i in range(len(xTest)):
        error += abs(hypo[i] - yTest[i])/max(yTest[i],hypo[i])
        print("GDP Predict Result:",float('%.1f' % hypo[i]),"GDP Real Label:",float('%.1f' % yTest[i]))
    error /= len(xTest)
    #k = model.predict(xTest)
    #for i in range(len(xTest)):
        #print(format(k[i],'.3f'),format(yTest[i],'.3f'),abs(k[i] - yTest[i])/yTest[i])
    return error

# Loop to find appropriate c value and gamma value
import KFoldValidation as KF
dataset = KF.KFold(X_train,Y_train)

averageError = 0
averageTrainError = 0
for i in range(1, 11):
    X_train, Y_train, X_validation, Y_validation = dataset.spilt(i)
    c_rbf = 90000
    c_poly = 1000
    gamma = np.power(1.3,-5.)
    rbf = svm.SVR(C= c_rbf,kernel='rbf',gamma = gamma,tol = 1e-9)
    poly = svm.SVR(C = c_poly,kernel='poly',gamma = 'auto',tol = 1e-8)
    rbf.fit(X_train,Y_train)
    poly.fit(X_train,Y_train)
    error = validation(rbf, X_validation, Y_validation)
    trainError = validation(rbf,X_train,Y_train)
    averageError += error
    averageTrainError += trainError
    print(error)

print("averageTrainScore:",averageTrainError / 10)
print("averageScore:",averageError / 10)
gamma = np.power(1.3, -5.)
rbf = svm.SVR(C=90000, kernel='poly', gamma=gamma, tol=1e-9)
rbf.fit(X_train, Y_train)
print('**Test Validation Start!**')
validation(rbf,X_test,Y_test)
# Kofu

X = np.asarray([[790.85,37.13,1,51.88,11,2,193127,35.6623,138.5682]])
Dusseldorf = [[1133,53.82,1,52.56,17,2,598686,51.2251964,6.7737511]]

# GDP = 67393 in 2012
#print("KofuGDP:",rbf.predict(X))
#print("testScore:",validation(rbf,X_test,Y_test))


# After selecting c and checking with random set of dataset, the error rate is about 50% which is good in some degree
# Now I will try the regression MLP
