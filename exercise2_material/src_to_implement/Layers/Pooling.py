import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    """
    最大池化层 (Max-Pooling Layer)。
    仅支持 2D 输入 (b, c, y, x)。
    使用 'valid' 填充 (无零填充)。
    """

    def __init__(self, stride_shape, pooling_shape):
        """
        构造函数。
        :param stride_shape: 步长 (tuple/int)，定义池化窗口的移动距离 (sy, sx)。
        :param pooling_shape: 池化窗口大小 (tuple)，定义窗口的尺寸 (py, px)。
        """
        super().__init__()
        self.trainable = False

        # 处理步长参数
        if isinstance(stride_shape, int):
            self.stride = (stride_shape, stride_shape)
        elif isinstance(stride_shape, tuple) and len(stride_shape) == 2:
            self.stride = stride_shape
        else:
            raise ValueError("stride_shape must be an int or a 2-element tuple for 2D pooling.")

        # 处理池化窗口尺寸
        if isinstance(pooling_shape, int):
            self.pooling_size = (pooling_shape, pooling_shape)
        elif isinstance(pooling_shape, tuple) and len(pooling_shape) == 2:
            self.pooling_size = pooling_shape
        else:
            raise ValueError("pooling_shape must be an int or a 2-element tuple for 2D pooling.")

        # 用于反向传播：存储 Max 值的位置索引
        self.max_indices = None
        # 用于反向传播：存储输入张量的形状
        self.input_shape = None

    def forward(self, input_tensor):
        """
        执行最大池化前向传播。
        输入布局：(b, c, y, x)

        :param input_tensor: 输入数据 (NumPy 数组)
        :return: 输出特征图
        """
        B, C, H_in, W_in = input_tensor.shape
        self.input_shape = input_tensor.shape

        P_H, P_W = self.pooling_size
        S_H, S_W = self.stride

        # 1. 计算输出形状 (使用 'valid' 填充，不填充)
        # H_out = floor((H_in - P_H) / S_H) + 1
        # W_out = floor((W_in - P_W) / S_W) + 1
        H_out = int(np.floor((H_in - P_H) / S_H)) + 1
        W_out = int(np.floor((W_in - P_W) / S_W)) + 1

        output_shape = (B, C, H_out, W_out)
        output_tensor = np.zeros(output_shape)

        # 2. 存储最大值索引 (用于反向传播)
        # 索引张量与输入形状相同，存储的是最大值在对应窗口内的相对位置
        # 为了高效，我们可以将输入展平为 [B * C, H_in, W_in] 进行处理
        # 存储的索引是 (B, C, H_out, W_out, P_H, P_W) 中的最大值在 (P_H, P_W) 中的位置

        # 我们使用一个与输出形状相同的张量来存储最大值在 H_in, W_in 上的**绝对索引**
        # 形状：(B, C, H_out, W_out, 2)
        self.max_indices = np.zeros((B, C, H_out, W_out, 2), dtype=int)

        # 3. 滑动窗口执行 Max-Pooling
        for h_out in range(H_out):
            for w_out in range(W_out):
                # 计算当前窗口的起始和结束坐标
                h_start = h_out * S_H
                w_start = w_out * S_W
                h_end = h_start + P_H
                w_end = w_start + P_W

                # 提取当前窗口的输入数据 (所有 Batch 和 Channel)
                window = input_tensor[:, :, h_start:h_end, w_start:w_end]

                # 找到窗口内的最大值
                # 在窗口的 H/W 维度上求 max (轴 2 和 3)
                max_value = np.max(window, axis=(2, 3))
                output_tensor[:, :, h_out, w_out] = max_value

                # 4. 存储 Max 值的索引 (关键步骤)
                # 计算最大值在窗口内的相对索引
                max_index_flat = np.argmax(window.reshape(B, C, -1), axis=2)

                # 将展平的相对索引转换回 (P_H, P_W) 上的 2D 相对索引
                rel_row_index = max_index_flat // P_W
                rel_col_index = max_index_flat % P_W

                # 计算最大值在原始输入张量上的**绝对索引** (H_in, W_in)
                abs_row_index = rel_row_index + h_start
                abs_col_index = rel_col_index + w_start

                self.max_indices[:, :, h_out, w_out, 0] = abs_row_index
                self.max_indices[:, :, h_out, w_out, 1] = abs_col_index

        return output_tensor

    def backward(self, error_tensor):
        """
        执行最大池化反向传播。
        只将梯度传递给前向传播中被选为最大值的那个位置。

        :param error_tensor: 下一层传回的误差梯度 (dL/dY)
        :return: 传给上一层的误差梯度 (dL/dX)
        """
        if self.max_indices is None or self.input_shape is None:
            raise ValueError("Backward pass requires max_indices and input_shape to be set by forward pass.")

        B, C, H_in, W_in = self.input_shape
        B, C, H_out, W_out = error_tensor.shape

        # 1. 初始化传给上一层的梯度张量 (dL/dX)
        pre_layer_error_tensor = np.zeros(self.input_shape)

        # 2. 将误差梯度 (dL/dY) 散播到原始输入的对应位置
        # 遍历输出梯度张量 (dL/dY) 的每个元素
        for b in range(B):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        # 获取 Max 值在原始输入张量上的绝对索引
                        abs_h = self.max_indices[b, c, h_out, w_out, 0]
                        abs_w = self.max_indices[b, c, h_out, w_out, 1]

                        # 只有被选中的 Max 位置才能接收梯度
                        # pre_layer_error_tensor[b, c, abs_h, abs_w] += error_tensor[b, c, h_out, w_out]

                        # 优化：可以使用 NumPy 的高级索引来加速，但为了清晰，我们保持循环
                        # 如果同一个位置是多个窗口的最大值，梯度应该累加（但 Max Pooling 通常不会重叠太多）

                        # 在当前最大值位置累加梯度
                        pre_layer_error_tensor[b, c, abs_h, abs_w] += error_tensor[b, c, h_out, w_out]

        return pre_layer_error_tensor