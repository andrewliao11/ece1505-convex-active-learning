import scipy 
import numpy as np
from sklearn.svm import SVC


class SVMLearner(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.svm = SVC(probability=True)
        self.svm.fit(X, y)

    def predict_proba(self, X):
        prob = self.svm.predict_proba(X)
        return prob
    
    def cal_uncertainty(self, prob, sigma, K):
        entropy = scipy.stats.entropy(prob, axis=1)
        return sigma - (sigma - 1) * entropy / np.log(K)
