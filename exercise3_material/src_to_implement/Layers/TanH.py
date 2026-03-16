import numpy as np

from Layers.Base import BaseLayer


class TanH(BaseLayer):
    def __init__(self):
        # 用于存储 forward 阶段计算出的激活值 (a)
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        """
        f(x) = tanh(x)
        """
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        """
        f'(x) = 1 - f(x)^2
        """
        # 利用存储的激活值计算梯度：gradient = error_tensor * (1 - a^2)
        gradient = error_tensor * (1 - np.power(self.activations, 2))
        return gradient