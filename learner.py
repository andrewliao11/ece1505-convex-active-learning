import numpy as np
from sklearn.svm import SVC
from scipy.optimize import least_squares


class Learner:
    
    def fit(self, X, Y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()
    
    def predict_proba(self, X):
        raise NotImplementedError()


class SVMLearner(Learner):
    def __init__(self, K):
        #TODO: It does not seem that fix the random state can remove the randomness
        self.svm = SVC(probability=True, random_state=123)
        self.K = K

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict_proba(self, X):
        prob = self.svm.predict_proba(X)
        
        seen_classes = self.svm.classes_
        all_classes = np.arange(self.K)
        if len(seen_classes) < len(all_classes):
            n_unseen = len(all_classes) - len(seen_classes)
            new_prob = np.zeros([len(prob), self.K]) + 1. / self.K
            for i, c in enumerate(seen_classes):
                new_prob[:, c] = prob[:, i] * (1 - n_unseen / self.K)
            prob = new_prob

        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return prob.argmax(1)
    
    def cal_ece(self, prob, true):
        ece = cal_ece(prob, true)
        return ece

    def calibrate_bin(self, prob, y):

        ece = cal_ece(prob, y)
        print("Before calibration: {:.2f}".format(ece))
        prob = perfect_calibrate(prob, y)
        ece = cal_ece(prob, y)
        print("After calibration: {:.2f}".format(ece))
        return prob


#https://github.com/markus93/NN_calibration/blob/master/scripts/utility/evaluation.py#L130-L154
def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin
  

def cal_ece(prob, true, bin_size = 0.1):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    
    conf = prob.max(1)
    pred = prob.argmax(1)
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece


def perfect_calibrate(prob, true, bin_size = 0.1):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    
    conf = prob.max(1)
    pred = prob.argmax(1)
    correct = pred == true
    new_prob = np.zeros_like(prob)

    def f(prob, pred, desired_acc): 
        def _f(tau): 
            logits = np.log(prob) / tau 
            z = np.exp(logits).sum() + 1e-8 
            new_prob = np.exp(logits) / z 
            return new_prob[pred] - desired_acc 
        return _f


    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        low = conf_thresh - bin_size
        high = conf_thresh
        acc, avg_conf, len_bin = compute_acc_bin(low, high, conf, pred, true)
        mask = np.logical_and(conf >= low, conf <= high)
        if mask.sum() > 0 and acc > 1 / prob.shape[1]:
            for i in np.where(mask)[0]:
                tau = least_squares(f(prob[i], pred[i], acc), np.array([1.]), bounds=[0.01, 10.])
                logits = np.log(prob[i]) / tau.x
                new_prob[i] = np.exp(logits) / np.exp(logits).sum()
                assert new_prob[i].argmax() == prob[i].argmax(), ipdb.set_trace()
        else:
            new_prob[mask] = prob[mask]
            
    return new_prob
