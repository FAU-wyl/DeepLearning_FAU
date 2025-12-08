import copy


class NeuralNetwork:
    def __init__(self, optimizer):
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
        """
        Target:
            Implement a method backward starting from the loss  layer passing it the label  tensor
            for the current input and propagating it back through the network.
        """
        label_tensor = self._current_label_tensor

        # run Loss.backward() here
        error_tensor = self.loss_layer.backward(label_tensor)

        # 3. 逐层反向传播 (从后往前遍历 layers)
        # reversed(self.layers) 反转列表
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        """
        Target:
            If  the  layer  is  trainable,  it  makes  a
            deep copy  of  the  neural  network’s  optimizer  and  sets  it  for  the  layer  by  using  its
            optimizer property.  Both, trainable and non-trainable layers, are then appended to the
            list layers.
        """
        if layer.trainable:
            # Note: We will implement optimizers that have an internal state in the upcoming exercises,
            # which makes copying of the optimizer object necessary.
            layer.optimizer = copy.deepcopy(self.optimizer)
        # 将层加入列表
        self.layers.append(layer)

    def train(self, iterations):
        """
        Target:
            implement a convenience method train(iterations), which trains the network
            for iterations and stores the loss for each iteration.
        """
        for i in range(iterations):
            # iterations belongs to Class NeuralNetwork

            # Forward pass
            loss_value = self.forward()

            # Store loss
            self.loss.append(loss_value)

            # Backward pass
            self.backward()

    def test(self, input_tensor):
        """
        Target：
            Finally implement a convenience method test(input  tensor) which propagates the in-
            put  tensor through the network and returns the prediction of the last layer.  For clas-
            sification tasks we typically query the probabilistic output of the SoftMax layer.

        """
        # 只需要前向传播通过所有普通层 (不包括 loss_layer)
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        return output


