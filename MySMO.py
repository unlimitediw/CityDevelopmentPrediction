import numpy as np


class SVM_HAND(object):
    def __init__(self, C, XSample, YSample, tolerance = .1, sigma = 3, kernel = 'rbf'):
        self.XSample = XSample
        self.YSample = YSample
        self.C = C
        self.alpha = np.zeros(YSample.shape)
        self.b = 0
        self.sigma = sigma
        self.kernel = kernel
        self.m = len(YSample)
        self.SMO(XSample, YSample, C,tolerance = tolerance)

    def Kernel(self, xi, xj):
        '''

        :param xi: np.ndarray
        :param xj: np.ndarray
        :param sigma: the lower the sigma, the sharper the model
        :param kernel: type of kernel
        :return: gaussian kernel of <xi,xj>
        '''
        if self.kernel == 'linear':
            return xi.dot(xj)
        if self.kernel == 'rbf':
            l2_square = np.sum(np.square(xi - xj), axis=-1)
            k = -np.float64(l2_square/self.sigma ** 2)
            return np.exp(k)
        if self.kernel == 'polynomial':
            return (1 + xi.dot(xj)) ** 2

    def predict(self,x):
        kernel = self.Kernel(self.XSample, x)
        result = np.sum(self.alpha * self.YSample * kernel) + self.b
        return 1 if result >= 0 else -1

    # function 1
    def Hypo(self, x):
        '''
        :param alpha: the alpha i weight for sample point
        :param yi: yi for sample point
        :param b: threshold for solution
        :param xi: xi for sample point
        :param xj: xj for input data
        :return: yj for predict result
        '''

        kernel = self.Kernel(self.XSample, x)
        result = np.sum(self.alpha * self.YSample * kernel) + self.b
        return result

    def LHBound(self, yi, yj, alphai, alphaj, C):
        '''

        :param yi: label for sample data
        :param yj: label for input data
        :param alphai: training alphai
        :param alphaj: training alphaj
        :param C:
        :return:
        '''
        if yi != yj:
            L = max(0, alphaj - alphai)
            H = min(C, C + alphaj - alphai)
        else:
            L = max(0, alphai + alphaj - C)
            H = min(C, alphai + alphaj)
        return L, H

    def Eta(self, xi, xj):
        return 2 * self.Kernel(xi, xj) - self.Kernel(xi, xi) - self.Kernel(xj, xj)

    def AlphaJUpdate(self, alphaJOld, yj, Ei, Ej, eta, H, L):
        alphaJNew = alphaJOld - yj * (Ei - Ej) / eta
        if alphaJNew > H:
            return H
        elif alphaJNew < L:
            return L
        else:
            return alphaJNew

    def AlphaIUpdate(self, alphaIOld, alphaJOld, alphaJNew, yi, yj):
        return alphaIOld + yi * yj * (alphaJOld - alphaJNew)

    def BUpdate(self, bOld, Ei, Ej, xi, xj, yi, yj, alphaINew, alphaJNew, alphaIOld, alphaJOld):
        b1 = bOld - Ei - yi * (alphaINew - alphaIOld) * self.Kernel(xi, xi) - yj * (
                alphaJNew - alphaJOld) * self.Kernel(xi, xj)
        if 0 < alphaINew < self.C:
            return b1
        b2 = bOld - Ej - yi * (alphaINew - alphaIOld) * self.Kernel(xi, xj) - yj * (
                alphaJNew - alphaJOld) * self.Kernel(xj, xj)
        if 0 < alphaJNew < self.C:
            return b2
        else:
            return (b1 + b2) / 2

    def SMO(self, XSample, YSample, C, tolerance=.1, maxPasses=30):
        '''
        :param C:
        :param tolerance:
        :param maxPasses:
        :param XSample:
        :param YSample:
        :param X:
        :param sigma:
        :param kernelT:
        :return: alpha
        '''
        passes = 0
        self.m = len(YSample)
        while passes < maxPasses:
            num_changed_alphas = 0
            for i in range(self.m):
                # Calculate Ei using f(xi) - y(i)
                hypoI = self.Hypo(self.XSample[i])
                Ei = hypoI - YSample[i]
                if (YSample[i] * Ei < -tolerance and self.alpha[i] < C) or (
                        YSample[i] * Ei > tolerance and self.alpha[i] > 0):
                    # Randomly select a j != i
                    j = i
                    while i == j:
                        j = np.random.randint(1, self.m)
                    # Calculate Ej using f(xj) - y(j)
                    hypoJ = self.Hypo(self.XSample[j])
                    Ej = hypoJ - YSample[j]
                    # Memo old alpha
                    alphaIOld = self.alpha[i]
                    alphaJOld = self.alpha[j]
                    # Compute L and H
                    L, H = self.LHBound(YSample[i], YSample[j], alphaIOld, alphaJOld, C)
                    if L == H:
                        continue
                    # Compute eta
                    eta = self.Eta(XSample[i], XSample[j])
                    if eta >= 0:
                        continue
                    # Compute and clip new value for alphaj using
                    self.alpha[j] = self.AlphaJUpdate(alphaJOld,YSample[j],Ei,Ej,eta,H,L)
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L
                    if abs(self.alpha[j] - alphaJOld) < 10 ^ -5:
                        continue
                    # Determine value for alphai
                    self.alpha[i] = self.AlphaIUpdate(alphaIOld,alphaJOld,self.alpha[j],YSample[i],YSample[j])
                    # Compute b
                    self.b = self.BUpdate(self.b, Ei, Ej, XSample[i], XSample[j], YSample[i], YSample[j], self.alpha[i],
                                     self.alpha[j],
                                     alphaIOld, alphaJOld)
                    num_changed_alphas += 1
            #print(num_changed_alphas,passes)
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
