import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    """
    Flatten layers reshapes the multi-dimensional input to a one dimensional feature vector.
    """

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape[1:]

        # reshapes and returns the input tensor.
        output_tensor = input_tensor.reshape(input_tensor.shape[0], -1)

        return output_tensor

    def backward(self, error_tensor):
        """
        将下一层传回的梯度张量（已展平）重新整形回其原始的多维形状。
        """
        if self.input_shape is None:
            # 理论上不应该发生，除非没有执行 forward
            raise ValueError("Flatten layer's input_shape is not set. Run forward pass first.")

        # 1. 重新构造目标形状
        # 目标形状是 (Batch_Size, C, H, W)
        target_shape = (error_tensor.shape[0],) + self.input_shape

        # 2. 执行反向展平操作
        # 将展平的梯度重新变回多维的形状
        reshaped_error_tensor = error_tensor.reshape(target_shape)
        return reshaped_error_tensor