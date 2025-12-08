import numpy as np
import math

class Constant:
    """用一个常数值初始化权重。"""
    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        """返回一个 Constant 初始化的 NumPy 数组。"""
        # 使用 np.full 创建一个指定形状和常数值的数组
        return np.full(weights_shape, self.constant_value)

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # the uniform distribution is the interval [0; 1).
        return np.random.uniform(0.0, 1.0, size=weights_shape)

class Xavier:
    """
    Zero-mean Gaussian: N (0, σ)
    sigma = sqrt(2 / (fan_in + fan_out))
    """
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(loc=0.0, scale=std, size=weights_shape)

class He:
    """
    Zero-mean Gaussian: N (0, σ)
    Standard deviation of weights determined by size of previous layer only
    sigma = sqrt(2 / fan_in)

    """
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):

        std = math.sqrt(2.0 / fan_in)
        return np.random.normal(loc=0.0, scale=std, size=weights_shape)