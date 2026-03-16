import numpy as np

from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        # 用于存储 forward 阶段计算出的激活值 (a)
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        """
        f(x) = 1 / (1 + exp(-x))
        """
        self.activations = 1.0 / (1.0 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        """
        f'(x) = f(x) * (1 - f(x))
        """
        # 利用存储的激活值计算梯度：gradient = error_tensor * a * (1 - a)
        gradient = error_tensor * self.activations * (1.0 - self.activations)
        return gradient