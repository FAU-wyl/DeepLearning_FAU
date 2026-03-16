import copy


class NeuralNetwork():
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        # 1. An optimizer object received upon construction as the First argument.
        self.optimizer = optimizer

        # 2. A list loss which will contain the loss value for each iteration after calling train.
        self.loss = []

        # 3. A list layers which will hold the architecture
        self.layers = []

        # 4. a member data  layer, which will provide input data and labels
        self.data_layer = None

        # 5. CrossEntropyLoss,a member loss layer referring to the special layer providing loss and prediction.
        self.loss_layer = None

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = 'train'

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        """
        Use this method to set the phase in the train and test methods.
        """
        self._phase = value
        for layer in self.layers:
            layer.phase = value

    def forward(self):
        #  the data layer provides an input tensor and a label tensor upon calling next() on it.
        self.input_tensor, self.label_tensor = self.data_layer.next()

        self._current_label_tensor = self.label_tensor

        current_tensor = self.input_tensor

        # Forward pass through all layers in the network
        for layer in self.layers:
            # run FullyConnected.forward() here
            current_tensor = layer.forward(current_tensor)

        loss_val = self.loss_layer.forward(current_tensor, self.label_tensor)
        return loss_val

    def backward(self):
        label_tensor = self._current_label_tensor

        # run Loss.backward() here
        error_tensor = self.loss_layer.backward(label_tensor)

        # reversed(self.layers) 反转列表
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)


        if hasattr(layer, 'initialize'):
            layer.initialize(self.weights_initializer, self.bias_initializer)

    def calculate_regularization_loss(self):
        """
        Refactor the NeuralNetwork class to add the regularization loss to the data loss.  Use
        the method norm(weights) to get the regularization loss inside all layers (Fully Connected, Convolution and RNN) and sum it up.
        """
        reg_loss = 0
        for layer in self.layers:
            if hasattr(layer, 'optimizer') and layer.optimizer is not None:
                if layer.optimizer.regularizer is not None:
                    reg_loss += layer.optimizer.regularizer.norm(layer.weights)
        return reg_loss

    def train(self, iterations):
        self.phase = 'train'

        for i in range(iterations):
            loss_value = self.forward()

            # loss + regularization_loss
            total_loss = loss_value + self.calculate_regularization_loss()

            self.loss.append(total_loss)
            self.backward()

    def test(self, input_tensor):
        self.phase = 'test'

        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        return output
