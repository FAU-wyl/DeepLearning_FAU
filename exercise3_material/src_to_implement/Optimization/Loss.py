import numpy as np


class CrossEntropyLoss:
    def __init__(self):

        self.prediction_tensor = None  # 用于存储输入的预测值 (Y_hat)

    def forward(self, prediction_tensor, label_tensor):
        """
        Target:
        computes the Loss value according the CrossEntropy Loss formula
        accumulated over the batch.
        """
        self.prediction_tensor = prediction_tensor

        # Add small epsilon for numerical stability to avoid log(0)
        eps = np.finfo(float).eps  # approximately 2.22e-16
        # cross entropy loss = -sum(y*log(y_hat + eps))

        loss = -np.sum(label_tensor * np.log(prediction_tensor + eps))
        return loss

    def backward(self, label_tensor):
        """
        Target:
        returns the error  tensor for the previous layer. The  backpropagation  starts  here,  hence  no  error  tensor  is  needed.
        Instead, we need the label  tensor.
        """
        eps = np.finfo(float).eps  # approximately 2.22e-16
        # E = -y/(y_hat + eps)
        error_tensor = - (label_tensor / (self.prediction_tensor + eps))
        return error_tensor
