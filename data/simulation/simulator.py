import numpy as np
from sklearn.datasets import make_moons, make_blobs



class Simulator():
    def __init__(self, data_type: str, noise: float):
        assert data_type in ["moon", "blob"]
        self.npr = np.random.RandomState(123)
        self.data_type = data_type
        self.noise = noise

    def simulate(self, n, input_dim):
        if self.data_type == "moon":
            assert input_dim == 2
            # w/ gaussian noise
            X, y = make_moons(n, noise=self.noise)
        if self.data_type == "blob":
             X, y = make_blobs(
                    n_samples=n, 
                    centers=5, 
                    n_features=input_dim,
                    random_state=self.npr
                )

        # Ensure the data is random shuffled
        idx = np.arange(n)
        self.npr.shuffle(idx)
        X = X[idx]
        y = y[idx]

        return X, y
