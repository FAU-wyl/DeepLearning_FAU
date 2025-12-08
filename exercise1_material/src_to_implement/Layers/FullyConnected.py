import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        """
        全连接层构造函数
        :param input_size: 输入特征的数量
        :param output_size: 输出特征的数量
        """
        # 1. First,  call  its  super-constructor.
        super().__init__()

        # 2. Set  the  inherited  member  trainable  to  True,  as  this layer  has  trainable  parameters.
        self.trainable = True

        # 3. Initialize weights uniformly random in [0, 1)
        # Shape: [input_size + 1, output_size], where +1 is for bias
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))

        self._optimizer = None
        self.input_tensor = None  # 用于存储前向传播的输入（含 Bias）
        self._gradient_weights = None  # 用于存储计算出的梯度

    def forward(self, input_tensor):

        # Get batch_size, represents the number of inputs processed simultaneously.
        batch_size = input_tensor.shape[0]

        # input_tensor shape: (batch_size, input_size) -> (batch_size, input_size + 1)
        bias_column = np.ones((batch_size, 1))

        # Add bias column (column of 1) to input_tensor(在输入矩阵右侧拼接一列全 1）
        self.input_tensor = np.concatenate((input_tensor, bias_column), axis=1)

        # Y = X * W
        output_tensor = np.dot(self.input_tensor, self.weights)
        # returns a tensor that serves as the input  tensor  for  the  next  layer.
        return output_tensor

    # Add a setter and getter property optimizer which sets and returns the protected member optimizer for this layer.
    @property
    def optimizer(self):
        """ Optimizer 的 Getter """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """ Optimizer 的 Setter """
        self._optimizer = optimizer

    def backward(self, error_tensor):
        """
        :param error_tensor: 下一层传回来的误差梯度
        :return: pre_layer_error_tensor 传给上一层的误差梯度
        """

        """
        Target：
            Use  the  method  calculate  update(weight  tensor,  gradient  tensor)  of  your  optimizer
            in  your  backward  pass,  in  order  to  update  your  weights.  
        """
        # Compute gradient of weights(dL/dW = dL/dy * dy/dW = X * dL/dy， dy/dW = X
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        # update if  the optimizer is  not set.
        if self._optimizer is not None:  # w‘ = w - learning_rate * w_gradient
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        """ 
        Target： 
            returns a tensor that serves as the error tensor for the previous layer.
            ps: prev_error_tensor includes the gradient of Bias (the latest column)，
            The output of pre_layer  doesn't have bias，so it dosen't need gradient of Bias.
        
        Function: 
            Error_pre = Error * Weights.T
        """
        pre_layer_error_tensor = np.dot(error_tensor, self.weights.T)
        return pre_layer_error_tensor[:, :-1]

    @property
    def gradient_weights(self):
        """
        For  future  reasons  provide  a  property  gradient  weights  which  returns  the  gradient  with  respect  to  the  weights,  after  they  have
        been calculated in the backward-pass.  These properties are accessed by the unit tests
        and are therefore also important to pass the tests!
        """
        return self._gradient_weights


