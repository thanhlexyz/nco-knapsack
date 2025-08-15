import numpy as np
import torch

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-8, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean    = np.zeros(shape, "float64")
        self.var     = np.ones(shape, "float64")
        self.count   = epsilon
        self.epsilon = epsilon

    def update(self, x):
        # custom wrap scalar reward to vector one-dimensional
        if type(x) in [float, np.float64]:
            x = np.array([x])
        # end custom
        """Updates the mean, var and count from a batch of samples."""
        batch_mean  = np.mean(x, axis=0)
        batch_var   = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

    def normalize_torch(self, x):
        return (x - torch.tensor(self.mean, dtype=torch.float)) / \
                torch.sqrt(torch.tensor(self.var, dtype=torch.float) + self.epsilon)
