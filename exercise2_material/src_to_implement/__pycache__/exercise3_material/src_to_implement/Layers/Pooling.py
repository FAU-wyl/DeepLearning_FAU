import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False

        if isinstance(stride_shape, int):
            self.stride = (stride_shape, stride_shape)
        elif isinstance(stride_shape, tuple) and len(stride_shape) == 2:
            self.stride = stride_shape

        if isinstance(pooling_shape, int):
            self.pooling_size = (pooling_shape, pooling_shape)
        elif isinstance(pooling_shape, tuple) and len(pooling_shape) == 2:
            self.pooling_size = pooling_shape

        self.max_indices = None
        self.input_shape = None

    def forward(self, input_tensor):
        """
        Requires:
            returns a tensor that serves as the input  tensor  for  the  next  layer.  Hint:  Keep  in  mind  to  store  the  correct  information
            necessary for the backward pass.
        """
        B, C, H_in, W_in = input_tensor.shape
        self.input_shape = input_tensor.shape

        P_H, P_W = self.pooling_size
        S_H, S_W = self.stride

        # 1. Calculate the output shape
        H_out = int(np.floor((H_in - P_H) / S_H)) + 1
        W_out = int(np.floor((W_in - P_W) / S_W)) + 1

        output_shape = (B, C, H_out, W_out)
        output_tensor = np.zeros(output_shape)

        # 2.store  the  correct  information necessary for the backward pass.
        self.max_indices = np.zeros((B, C, H_out, W_out, 2), dtype=int)

        # 3. Max-Pooling
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * S_H
                w_start = w_out * S_W
                h_end = h_start + P_H
                w_end = w_start + P_W

                # 提取当前窗口的输入数据 (所有 Batch 和 Channel)
                window = input_tensor[:, :, h_start:h_end, w_start:w_end]

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
                        abs_h = self.max_indices[b, c, h_out, w_out, 0]
                        abs_w = self.max_indices[b, c, h_out, w_out, 1]
                        pre_layer_error_tensor[b, c, abs_h, abs_w] += error_tensor[b, c, h_out, w_out]
        return pre_layer_error_tensor
