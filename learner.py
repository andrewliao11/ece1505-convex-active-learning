from sklearn.svm import SVC


class SVMLearner(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        #TODO: It does not seem that fix the random state can remove the randomness
        self.svm = SVC(probability=True, random_state=123)
        self.svm.fit(X, y)

    def predict_proba(self, X):
        prob = self.svm.predict_proba(X)
        return prob
    