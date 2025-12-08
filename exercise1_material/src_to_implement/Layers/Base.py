class BaseLayer:
    def __init__(self):
        """
        trainable: 布尔值，用于区分该层是否包含可训练的参数（如权重）。
                   默认为 False。
        """
        self.trainable = False  # be used  to  distinguish trainable from non-trainable layers.

        # 用于存储该层的权重参数（如 weights, bias），在后续实现优化器更新权重时有用
        self.weights = []  # Optionally, you can add other members like a default weights parameter, which might come in handy.

    def forward(self, input_tensor):
        raise NotImplementedError("Subclasses must implement the forward method")

    def backward(self, error_tensor):
        raise NotImplementedError("Subclasses must implement the backward method")
