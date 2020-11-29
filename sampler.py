import time
from enum import Enum
import scipy 
import cvxpy as cp
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering
from itertools import combinations 
import ipdb


class Sampler():
    def __init__(self, X, Y, labeled_mask):
        self.X = X
        self.Y = Y
        self.labeled_mask = labeled_mask
        
    def sample(self, n, **kwargs):
        raise NotImplementedError


class CVXSampler(Sampler):
    def __init__(self, X, Y, labeled_mask, params):
        Sampler.__init__(self, X, Y, labeled_mask)
        self.K = params.K          # Number of classes
        self.confidence_type = params.confidence_type
        self.clustering_type = params.clustering_type
        self.diversity_type = params.diversity_type
        self.sigma = params.sigma
        self.alpha = params.alpha  # Balance the two costs
        self.Z = None

    def cal_uncertainty(self, learner, X, labeled_mask):
        
        if self.confidence_type == "learner":
            prob = learner.predict_proba(X[~labeled_mask])
        elif self.confidence_type == "perfect_onehot_prob":
            pred = learner.predict(X[~labeled_mask])
            correct = pred == self.Y[~labeled_mask]
            n_unlabeled = (~labeled_mask).sum()
            prob = np.zeros([n_unlabeled, self.K])

            prob[correct, pred[correct]] = 1.
            prob[~correct] = 1/self.K
        elif self.confidence_type == "perfect_distributional_prob": 
            prob = learner.predict_proba(X[~labeled_mask])
            # calibrate the probability of each bin so as to retain the "distributional" information
            prob = learner.calibrate_bin(prob, self.Y[~labeled_mask])
        else:
            # None
            prob = np.ones([len(X[~labeled_mask]), self.K]) / self.K

        entropy = scipy.stats.entropy(prob, axis=1)
        uncertainty = self.sigma - (self.sigma - 1) * entropy / np.log(self.K)
        ece = learner.cal_ece(prob, self.Y[~labeled_mask])
        print("ECE: {:.2f}".format(ece))
        
        return uncertainty

    def cal_diversity(self, X, labeled_mask):
        s = rbf_kernel(X[~labeled_mask], X[labeled_mask])
        d = s.max() - s
        diversity = self.sigma - (self.sigma - 1) * d.min(1) / d.min(1).max()

        return diversity 

    def cal_dissimilarity_over_unlabeled_set(self, X, labeled_mask):

        if self.diversity_type == "optimal":
            labels = self.Y[~labeled_mask].reshape(-1,1)
            d = (labels.T != labels).astype(np.float)
        else:
            s = rbf_kernel(X[~labeled_mask])
            d = s.max() - s
        return d

    def sample(self, n, **kwargs):
        learner = kwargs["learner"]
        labeled_mask = self.labeled_mask
        X = self.X

        # Construct convex optimization problem
        n_unlabeled = (~labeled_mask).sum()
        if n_unlabeled <= n:
            return np.where(labeled_mask == 0)[0]

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
        
        self.Z = Z.value

        if self.clustering_type == "spectral":

            clustering_labels = SpectralClustering(n_clusters=n,
                assign_labels="discretize",
                random_state=0).fit(Z.value).labels_

            indices_by_cluster = {
                i: [] for i in range(n)
            }
            for i, label in enumerate(clustering_labels):
                indices_by_cluster[label].append(i)

            representatives = [inidices[0] for inidices in indices_by_cluster.values() if len(inidices) > 0]
        
        else:
            representatives = np.unique(Z.value.argmax(0))  # Index in unlabeled set

        idx_to_label = np.where(~labeled_mask)[0][representatives]  # Index in the whole set
        return idx_to_label


class RandomSampler(Sampler):
    def __init__(self, X, Y, labeled_mask, params):
        Sampler.__init__(self, X, Y, labeled_mask)
        self.npr = np.random.RandomState(123)

    def sample(self, n, **kwargs):
        unlabeled_idx = np.where(~self.labeled_mask)[0]
        return self.npr.choice(unlabeled_idx, n, replace=False)


class OptimalSampler(Sampler):
    def __init__(self, X, Y, labeled_mask, params):
        Sampler.__init__(self, X, Y, labeled_mask)
        self.npr = np.random.RandomState(123)
        self.params = params
        self.labeled_mask = labeled_mask

    def sample(self, n, **kwargs):
        learner = kwargs["learner"]
        test_x = kwargs["test_x"]
        test_y = kwargs["test_y"]

        unlabeled_idx = np.where(~self.labeled_mask)[0]

        # Iterate through every possible combination
        labeled_x = self.X[self.labeled_mask]
        labeled_y = self.Y[self.labeled_mask]
        learner.fit(labeled_x, labeled_y)
        curr_best = (learner.predict(labeled_x) == labeled_y).mean()
        best_batch = None
        for i, batch in enumerate(combinations(unlabeled_idx, n)):
            batch = list(batch)
            new_x = np.concatenate((labeled_x, self.X[batch]), axis=0)
            new_y = np.concatenate((labeled_y, self.Y[batch]), axis=0)
            learner.fit(new_x, new_y)
            acc = (learner.predict(test_x) == test_y).mean()

            if acc > curr_best:
                curr_best = acc
                best_batch = batch
            
            # Make tractable
            if i > 10000:
                break

        if best_batch:
            return best_batch
        else:
            return self.npr.choice(unlabeled_idx, n, replace=False)

