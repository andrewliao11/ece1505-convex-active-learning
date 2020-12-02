import numpy as np
from sklearn.datasets import make_moons, make_blobs


class Simulator():
    def __init__(self, data_type: str, noise: float, K: int, seed: int):
        assert data_type in ["moon", "blob"]
        self.seed = seed
        self.npr = np.random.RandomState(self.seed)
        self.data_type = data_type
        self.noise = noise
        self.K = K

    def simulate(self, n, input_dim):
        if self.data_type == "moon":
            assert input_dim == 2
            assert self.K == 2
            # w/ gaussian noise
            X, y = make_moons(n, noise=self.noise, random_state=self.seed)
        elif self.data_type == "blob":
            # not equally distributed
            _n = self.npr.randint(n, size=self.K)
            _n = (_n / (_n.sum() / n)).astype(np.int)
            idx = self.npr.choice(self.K)
            _n[idx] -= _n.sum() - n
            X, y = make_blobs(
                    n_samples=_n, 
                    #centers=self.K, 
                    n_features=input_dim,
                    #cluster_std=self.noise, 
                    random_state=self.seed
                )
        else:
            raise ValueError

        # Ensure the data is random shuffled
        idx = np.arange(n)
        self.npr.shuffle(idx)
        X = X[idx]
        y = y[idx]

        return X, y
