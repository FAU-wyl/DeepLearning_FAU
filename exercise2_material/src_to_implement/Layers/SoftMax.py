import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.probs = None  # 用于存储前向传播计算出的概率矩阵 (Y)

    def forward(self, x):
        """
        Compute softmax probabilities.
        x: shape (batch_size, num_classes)
        """

        # 为了避免 exp() 溢出，对每一行减去最大值（稳定计算）
        # subtract max value from each row
        shifted_input = x - np.max(x, axis=1, keepdims=True)

        # exponentiation
        exp_x = np.exp(shifted_input)

        # sum over classes (denominator of softmax)
        exp_values = np.sum(exp_x, axis=1, keepdims=True)
        """
        axis=0	Sum down the columns (get one value per column)
        axis=1	Sum across each row (get one value per row)
        keepdims=True means keep the reduced dimension as size 1.
        """



        # final probabilities
        self.probs = exp_x / exp_values

        """
        returns the estimated class probabilities 
        for each row representing an element of the batch.
        """
        return self.probs

    def backward(self, error_tensor):
        """
        error_tensor: 下一层传回来的误差梯度 (batch_size, num_classes)
        返回: 传给上一层的梯度
        """


        # Compute gradient of input
        # Gradient = Y * (E - sum(E * Y)) , Y is self.probs
        gradient = self.probs * (error_tensor - np.sum(error_tensor * self.probs, axis=1, keepdims=True))



        """      
        returns  a  tensor  that  serves  as
        the error  tensor for the previous layer.
        """
        return gradient
