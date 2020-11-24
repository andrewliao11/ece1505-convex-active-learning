import time
import scipy 
import cvxpy as cp
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


class Sampler():
    def __init__(self, X, labeled_mask):
        self.X = X
        self.labeled_mask = labeled_mask
        
    
    def sample(self, n, **kwargs):
        raise NotImplementedError


class CVXSampler(Sampler):
    def __init__(self, X, labeled_mask, K, sigma=5, alpha=1):
        Sampler.__init__(self, X, labeled_mask)
        self.K = K                  # Number of classes
        self.sigma = sigma
        self.alpha = alpha          # Balance the two costs

    def cal_uncertainty(self, learner, X, labeled_mask):
        prob = learner.predict_proba(X[~labeled_mask])
        entropy = scipy.stats.entropy(prob, axis=1)
        return self.sigma - (self.sigma - 1) * entropy / np.log(self.K)

    def cal_diversity(self, X, labeled_mask):
        s = rbf_kernel(X[~labeled_mask], X[labeled_mask])
        d = s.max() - s
        diversity = self.sigma - (self.sigma - 1) * d.min(1) / d.min(1).max()
        return diversity 

    def cal_dissimilarity_over_unlabeled_set(self, X, labeled_mask):
        s = rbf_kernel(X[~labeled_mask])
        d = s.max() - s
        return d

    def sample(self, n, learner):
        labeled_mask = self.labeled_mask
        X = self.X

        # Construct convex optimization problem
        n_unlabeled = (~labeled_mask).sum()
        uncertainty = self.cal_uncertainty(learner, X, labeled_mask)
        diversity = self.cal_diversity(X, labeled_mask)
        C = np.zeros([n_unlabeled, n_unlabeled])
        np.fill_diagonal(C, np.stack([uncertainty, diversity], 0).min(0))

        D = self.cal_dissimilarity_over_unlabeled_set(X, labeled_mask)
        ones = np.ones(n_unlabeled)
        Z = cp.Variable([n_unlabeled, n_unlabeled])

        q = 2         # q-norm
        objective_fn = cp.atoms.affine.trace.trace(D.T @ Z) + self.alpha * sum(cp.atoms.norm(C @ Z, q, axis=1))
        constraints = [Z >= 0, ones.T @ Z == ones.T]
        prob = cp.Problem(cp.Minimize(objective_fn), constraints)
        t1 = time.time()
        #TODO: It does not scale to large scale problem. Is there any way to speed up?
        print("Solving Problem")
        prob.solve()
        print("Takes {:.2f} sec".format(time.time() - t1))


        representatives = np.unique(Z.value.argmax(0))  # Index in unlabeled set
        idx_to_label = np.where(~labeled_mask)[0][representatives]  # Index in the whole set
        return idx_to_label, Z.value


class RandomSampler(Sampler):
    def __init__(self, X, labeled_mask):
        Sampler.__init__(self, X, labeled_mask)
        self.npr = np.random.RandomState(123)

    def sample(self, n, **kwargs):
        unlabeled_idx = np.where(~self.labeled_mask)[0]
        return self.npr.choice(unlabeled_idx, n, replace=False)
