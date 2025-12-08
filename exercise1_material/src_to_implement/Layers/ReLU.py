import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None  # 用于保存前向传播的输入

    def forward(self, input_tensor):

        self.input_tensor = input_tensor

        # ReLU: f(x) = max(0, x)
        ReLU = np.maximum(0, input_tensor)
        return ReLU

    def backward(self, error_tensor):
        """
        :param error_tensor: 从下一层传回来的误差梯度
        :return: 传给上一层的误差梯度
        """

        # 创建一个用于传递给上一层的梯度 tensor，初始化为 error_tensor
        pre_layer_error_tensor = error_tensor.copy()

        # Compute gradient:
        # derivative of ReLU is 1 when input > 0 and 0 when input <= 0
        pre_layer_error_tensor = np.where(self.input_tensor > 0, pre_layer_error_tensor, 0)
        return pre_layer_error_tensor
