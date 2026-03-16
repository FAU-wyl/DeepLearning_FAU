import numpy as np

"""
Both have to provide the methods calculate  gradient(weights) that
calculates a (sub-)gradient on the weights needed for the optimizer.  Additionally they have to
provide a method norm(weights), which is used to calculate the norm enhanced loss.
"""

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        """
        L2 = 1/2 * alpha * (||w||_2)^2
        L2 gradient: alpha * w
        """
        return self.alpha * weights

    def norm(self, weights):
        """
        alpha*(||w||_2)^2
        """
        return self.alpha * np.sum(np.square(weights))


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        """
        L1 gradient: alpha * sign(w)
        """
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        """
        alpha * ||w||_1
        """
        return self.alpha * np.sum(np.abs(weights))