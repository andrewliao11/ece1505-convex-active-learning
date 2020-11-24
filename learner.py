from sklearn.svm import SVC

class learner:
    
    def fit(self, X, Y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()
    
    def predict_proba(self, X):
        raise NotImplementedError()

class SVMLearner(learner):
    def __init__(self):
        #TODO: It does not seem that fix the random state can remove the randomness
        self.svm = SVC(probability=True, random_state=123)

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict_proba(self, X):
        prob = self.svm.predict_proba(X)
        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return prob.argmax(1)
    