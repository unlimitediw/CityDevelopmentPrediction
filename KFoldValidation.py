import numpy as np


class KFold(object):
    def __init__(self, X, Y, foldTotal=10):
        self.X = X
        self.Y = Y
        self.foldTotal = foldTotal
        self.spiltLength = len(self.Y) // foldTotal

    def spilt(self, foldTime):
        '''
        It will be a little not well distributed because there is a remain for len(self.Y) // foldTotal.
        But the remain will smaller than foldTotal and does not matter comparing with the large training set.
        :param foldTime: the counter of spilt operation
        :return: training data of input and label, validating
        '''

        validateStart = foldTime * self.spiltLength
        validateEnd = (foldTime + 1) * self.spiltLength
        trainX = np.concatenate((self.X[:validateStart], self.X[validateEnd:]))
        trainY = np.concatenate((self.Y[:validateStart], self.Y[validateEnd:]))
        validateX = self.X[validateStart:validateEnd]
        validateY = self.Y[validateStart:validateEnd]
        return trainX, trainY, validateX, validateY
