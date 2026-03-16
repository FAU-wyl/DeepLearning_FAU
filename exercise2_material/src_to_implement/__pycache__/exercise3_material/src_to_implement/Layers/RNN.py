import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Write a constructor, receiving the arguments (input  size, hidden  size, output  size).
        Here  input  size denotes  the  dimension  of  the  input  vector  while  hidden  size denotes
        the dimension of the hidden state.  Initialize the hidden state with all zeros.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc_h = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_y = FullyConnected(hidden_size, output_size)
        self.tanh = TanH()

        super().__init__()

        self.trainable = True
        self.memorize = False
        self.h_prev = None

        # BPTT
        self.input_stack = []
        self.fc_h_layers = []
        self.tanh_layers = []
        self.fc_y_layers = []


    @property
    def weights(self):
        return self.fc_h.weights

    @weights.setter
    def weights(self, value):
        if isinstance(value, np.ndarray) and value.size > 0:
            self.fc_h.weights = value

    """
    Implement  the  accessor  property  gradient  weights.  Here  the  weights  are  defined  as
    the weights which are involved in calculating the hidden  state as a stacked tensor.  E.g. if  the  
    hidden  state is  computed  with  a  single  Fully  Connected  layer,  which  receives  a stack  of 
    the  hidden  state  and  the  input  tensor,  the  weights  of  this  particular  Fully Connected 
    Layer, are the weights considered to be weights for the whole class.  In order to provide access to 
    the weights of the RNN layer, implement a getter and a setter with
    a property for the weights member.

    """

    @property
    def gradient_weights(self):
        return self.fc_h.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.fc_h.gradient_weights = value

    """
    
    To be able to reuse all regularizers, add the property to add an optimizer as
    optimizer and to calculate the loss caused by regularization as calculate  regularization  loss()
    as introduced in the regularization exercise. Finally add the method initialize(weights initializer,
    bias initializer) to use our initializers.
    """
    @property
    def optimizer(self):
        return self.fc_h.optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self.fc_h.optimizer = copy.deepcopy(opt)
        self.fc_y.optimizer = copy.deepcopy(opt)


    def forward(self, input_tensor):
        """
        Implement  a  method  forward(input  tensor)  which  returns  a  tensor  that  serves  as
        the  input  tensor  for  the  next  layer.   Consider  the  “batch”  dimension  as  the  “time”
        dimension of a sequence over which the recurrence is performed.  The first hidden  state for this
        iteration is all zero if the boolean member variable is False, otherwise restore the hidden state
        from the last iteration.  You can choose to compose parts of the RNN from other layers  you already implemented.
        """
        T = input_tensor.shape[0]
        if not self.memorize or self.h_prev is None:
            self.h_t = np.zeros((1, self.hidden_size))
        else:
            self.h_t = self.h_prev

        output = np.zeros((T, self.output_size))
        self.input_stack = []
        self.fc_h_layers = []
        self.tanh_layers = []
        self.fc_y_layers = []

        for t in range(T):
            x_t = input_tensor[t:t + 1, :]
            combined = np.concatenate((self.h_t, x_t), axis=1)
            self.input_stack.append(combined)

            h_linear = self.fc_h.forward(combined)
            self.h_t = self.tanh.forward(h_linear)
            y_t = self.fc_y.forward(self.h_t)

            output[t] = y_t

            self.fc_h_layers.append(copy.deepcopy(self.fc_h))
            self.tanh_layers.append(copy.deepcopy(self.tanh))
            self.fc_y_layers.append(copy.deepcopy(self.fc_y))

        self.h_prev = self.h_t
        return output

    def backward(self, error_tensor):
        """
        Implement  a  method  backward(error  tensor) which  updates  the  parameters  and  re-
        turns  the  error  tensor  for  the  next  layer.  Remember  that  optimizers  are  decoupled from
        our layers.  For the gradient calculation of some layers we need the input of the last forward pass
        at the respective point in time.  Be sure to save those values in the forward
        pass and set them when backpropagating through time.
        """
        T = error_tensor.shape[0]
        grad_input = np.zeros((T, self.input_size))
        grad_h_next = np.zeros((1, self.hidden_size))

        self.gradient_weights_fc_h = np.zeros_like(self.fc_h.weights)
        self.gradient_weights_fc_y = np.zeros_like(self.fc_y.weights)

        for t in reversed(range(T)):
            # 1. 输出层
            grad_h_from_output = self.fc_y_layers[t].backward(error_tensor[t:t + 1, :])
            grad_h_total = grad_h_from_output + grad_h_next

            # 2. 激活层与隐藏层
            grad_tanh = self.tanh_layers[t].backward(grad_h_total)
            grad_combined = self.fc_h_layers[t].backward(grad_tanh)

            # 3. 分离梯度
            grad_h_next = grad_combined[:, :self.hidden_size]
            grad_input[t] = grad_combined[:, self.hidden_size:]

            # 4. 累积梯度
            self.gradient_weights_fc_h += self.fc_h_layers[t].gradient_weights
            self.gradient_weights_fc_y += self.fc_y_layers[t].gradient_weights

        # 更新最终梯度
        self.fc_h.gradient_weights = self.gradient_weights_fc_h
        self.fc_y.gradient_weights = self.gradient_weights_fc_y

        if self.fc_h.optimizer:
            self.fc_h.weights = self.fc_h.optimizer.calculate_update(self.fc_h.weights, self.fc_h.gradient_weights)
        if self.fc_y.optimizer:
            self.fc_y.weights = self.fc_y.optimizer.calculate_update(self.fc_y.weights, self.fc_y.gradient_weights)
        return grad_input

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_h.initialize(weights_initializer, bias_initializer)
        self.fc_y.initialize(weights_initializer, bias_initializer)