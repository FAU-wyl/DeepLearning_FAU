import numpy as np

from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        """
            Implement the constructor for this layer which receives the argument channels.
            channels denotes the number of channels of the input  tensor  in both, the vector and the image case.

            initialize.  This layer has trainable parameters, so remember to set the
            inherited member trainable accordingly.
        """
        super().__init__()
        self.channels = channels
        self.trainable = True

        #  Initialize the bias β  and the weights γ
        self.weights = None
        self.bias = None
        self.initialize()

        self.optimizer = None
        self.bias_optimizer = None

        self.testing_phase = False

        # Moving Average
        self.moving_mean = None
        self.moving_var = None
        self.alpha = 0.8

        self.input_tensor = None
        self.x_hat = None
        self.mean = None
        self.var = None
        self.eps = 1e-10

    @property
    def phase(self):
        # 这是一个内部使用的 getter，虽然 NeuralNetwork 暂时没用到，但保持对称
        return 'test' if self.testing_phase else 'train'

    @phase.setter
    def phase(self, value):
        # 核心逻辑：将 'test'/'train' 字符串 转换为 内部的布尔开关
        if value == 'test':
            self.testing_phase = True
        else:
            self.testing_phase = False

    def initialize(self, weights_initializer=None, bias_initializer=None):
        #  Initialize the bias β  and the weights γ, according to the channels-size using the method
        """
        initialize ignores any assigned initializer and initializes always the weights γ with ones and the
        biases β with zeros, since you do not want the weights γ and bias β to have an impact at  the
        beginning  of  the  training.   Make  sure  you  optimize  the  weights  and  bias  in  the
        backward pass, but only if optimizers are defined.
        """
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def reformat(self, tensor):
        """
        Implement a method reformat(tensor) which receives the tensor that must be reshaped.
        Depending on the shape of the tensor, the method reformats the image-like tensor (with
        4  dimension)  into  its  vector-like  variant  (with  2  dimensions),  and  the  same  method  reformats the vector-like tensor into its image-like tensor variant.  Use this in the forward
        and the backward pass.
        """
        if len(tensor.shape) == 4:
            # 4D -> 2D
            # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
            B, C, H, W = tensor.shape
            reformatted = tensor.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            B, H, W, C = self.original_shape[0], self.original_shape[2], self.original_shape[3], self.channels
            reformatted = tensor.reshape(B, H, W, C).transpose(0, 3, 1, 2)
        return reformatted

    def forward(self, input_tensor):

        self.original_shape = input_tensor.shape

        if len(input_tensor.shape) == 4:
            x = self.reformat(input_tensor)
        else:
            x = input_tensor

        self.input_tensor = x

        if not self.testing_phase:
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)

            if self.moving_mean is None:
                self.moving_mean = self.mean
                self.moving_var = self.var
            else:

                """
                moving average is common:
                µ˜(k ) ≈ αµ˜(k−1) + (1 − α)µ[B(k)]
                σ˜2(k ) ≈ ασ˜2(k −1) + (1 − α)σ^2[B(k)]
                """
                self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * self.mean
                self.moving_var = self.alpha * self.moving_var + (1 - self.alpha) * self.var

            curr_mean = self.mean
            curr_var = self.var
        else:
            """
            Modify the Batch Normalization method forward(input  tensor) for the testing phase.
            Use  an  online  estimation  of  the  mean  and  variance.   Initialize  mean  and  variance  with
            the batch mean and the batch standard deviation of the first batch used for training.
            """
            curr_mean = self.moving_mean
            curr_var = self.moving_var

        # Normalization
        # x~ = (x - µ_B) / sqrt(variance^2 + eps)
        self.x_hat = (x - curr_mean) / np.sqrt(curr_var + self.eps)

        y_hat = self.weights * self.x_hat + self.bias

        if len(self.original_shape) == 4:
            return self.reformat(y_hat)
        return y_hat

    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:
            error_tensor_2d = self.reformat(error_tensor)
        else:
            error_tensor_2d = error_tensor

        # 1. gradient with respect to gamma and beta
        # grad_gamma = sum(error * x_hat)
        self.gradient_weights = np.sum(error_tensor_2d * self.x_hat, axis=0)
        # grad_beta = sum(error)
        self.gradient_bias = np.sum(error_tensor_2d, axis=0)

        # 2. gradient with respect to the input
        grad_input = compute_bn_gradients(error_tensor_2d, self.input_tensor, self.weights, self.mean, self.var)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        # 4. reformat
        if len(self.original_shape) == 4:
            return self.reformat(grad_input)
        return grad_input