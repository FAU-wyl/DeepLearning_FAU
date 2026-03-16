import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        """
        全连接层构造函数
        :param input_size: 输入特征的数量 (fan_in)
        :param output_size: 输出特征的数量 (fan_out)
        """
        super().__init__()
        self.trainable = True

        # 存储 fan_in 和 fan_out 以便初始化器使用
        self.fan_in = input_size
        self.fan_out = output_size

        # 初始默认使用 UniformRandom 初始化
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))

        self._optimizer = None
        self.input_tensor = None
        self._gradient_weights = None

    # ... [forward, optimizer getter/setter, gradient_weights property 保持不变] ...

    def forward(self, input_tensor):
        # Get batch_size, represents the number of inputs processed simultaneously.
        batch_size = input_tensor.shape[0]

        # input_tensor shape: (batch_size, input_size) -> (batch_size, input_size + 1)
        bias_column = np.ones((batch_size, 1))

        # Add bias column (column of 1) to input_tensor(在输入矩阵右侧拼接一列全 1）
        self.input_tensor = np.concatenate((input_tensor, bias_column), axis=1)

        # Y = X * W
        output_tensor = np.dot(self.input_tensor, self.weights)
        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        pre_layer_error_tensor = np.dot(error_tensor, self.weights.T)
        # 移除偏置项的梯度（最后一行）
        return pre_layer_error_tensor[:, :-1]

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    # =========================================================================
    # 🌟 新增/更新方法：initialize (纯 NumPy 实现)
    # =========================================================================
    def initialize(self, weights_initializer, bias_initializer):
        """
        使用提供的初始化器重新初始化主权重 W 和偏置 B。
        """

        # 1. 初始化 主权重 W
        # W 的形状是 (fan_in, fan_out)
        weights_shape_W = (self.fan_in, self.fan_out)

        # 调用初始化器，返回 NumPy 数组
        W = weights_initializer.initialize(weights_shape_W, self.fan_in, self.fan_out)

        # 2. 初始化 偏置 B
        # B 的形状是 (1, fan_out)
        weights_shape_B = (1, self.fan_out)

        # 偏置通常 fan_in 设为 1
        B = bias_initializer.initialize(weights_shape_B, 1, self.fan_out)

        # 3. 将 W 和 B 拼接回 self.weights (B 作为最后一行)
        self.weights = np.concatenate((W, B), axis=0)

