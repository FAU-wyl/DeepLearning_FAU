import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = np.float64(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        parameters:
            weight_tensor: 当前权重张量
            gradient_tensor: 对应的梯度张量

        Return:
             The updated weights after applying the gradient descent update.
        """
        # SGD Function: w‘ = w - learning_rate * gradient
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights
