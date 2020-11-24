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
    