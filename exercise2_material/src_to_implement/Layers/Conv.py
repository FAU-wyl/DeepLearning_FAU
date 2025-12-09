import numpy as np
import copy
from scipy.signal import correlate, convolve
import math

def _prepare_stride_and_padding(input_shape, stride_shape, kernel_shape):
    spatial_input_shape = np.array(input_shape[2:])
    spatial_kernel_shape = np.array(kernel_shape[2:])
    num_dims = len(spatial_input_shape)

    # stride
    if isinstance(stride_shape, int):
        stride = np.array([stride_shape] * num_dims)
    elif isinstance(stride_shape, (tuple,list)):
        stride = np.array(stride_shape)

    # Calculate Output Spatial Shape
    # Shape: floor((Input - 1) / Stride) + 1
    output_spatial_shape = np.floor((spatial_input_shape - 1) / stride) + 1
    output_spatial_shape = output_spatial_shape.astype(int)

    # compute padding from kernel size
    padding_needed = spatial_kernel_shape - 1
    padding_before = padding_needed // 2
    padding_after = padding_needed - padding_before

    padding = []
    padding.append((0, 0))  # Batch
    padding.append((0, 0))  # Channel
    for i in range(num_dims):
        padding.append((int(padding_before[i]), int(padding_after[i])))

    return stride, tuple(padding), output_spatial_shape

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        self.stride_shape = stride_shape

        '''
        For 1D, it has the convolution  shape [c,  m],  whereas for 2D, it has the shape [c, m, n], where c represents the number of input channels, and m, n represent the
        spatial extent of the filter kernel.
        '''
        self.convolution_shape = convolution_shape

        self.num_kernels = num_kernels  # num kernels is an integer value.
        self.is_2D = (len(convolution_shape) == 3)
        self.num_spatial_dims = 2 if self.is_2D else 1
        # weight Shape: (Num_Kernels, C_in, Spatial_Height, Spatial_Width)
        weight_shape = (num_kernels,) + tuple(convolution_shape)

        # Requirement: Initialize the parameters of this layer uniformly random in the range [0, 1).
        self.weights = np.random.uniform(0, 1, weight_shape)
        self.bias = np.random.uniform(0, 1, (num_kernels,))

        self._optimizer_weights = None
        self._optimizer_bias = None
        self.input_tensor = None
        self.padding = None
        self.stride = None

        self._gradient_weights = None
        self._gradient_bias = None

    # optimizer properties
    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer_weights = optimizer
        self._optimizer_bias = copy.deepcopy(optimizer)
########################################################
    """
    Requirement:Additionally  provide  two  properties:
        gradient  weights  and gradient  bias,  which return the gradient with respect to the
        weights and bias, after they have been calculated in the backward-pass.
    """
    @property
    def gradient_weights(self):
        return self._gradient_weights
    @property
    def gradient_bias(self):
        return self._gradient_bias
########################################################

    def initialize(self, weights_initializer, bias_initializer):
        C_in = self.convolution_shape[0]
        K_spatial = np.prod(self.convolution_shape[1:])
        fan_in = C_in * K_spatial
        fan_out = self.num_kernels * K_spatial

        weights_shape = self.weights.shape
        W = weights_initializer.initialize(weights_shape, fan_in, fan_out)
        self.weights = W.numpy() if hasattr(W, 'numpy') else W

        bias_shape = self.bias.shape
        B = bias_initializer.initialize(bias_shape, fan_in=1, fan_out=self.num_kernels)
        self.bias = B.numpy() if hasattr(B, 'numpy') else B

    def forward(self, input_tensor):
        """
        The input layout for 1D is defined in b, c, y order, for 2D in b, c, y, x order.  Here,
        b  stands  for  the  batch,  c  represents  the  channels  and  x,  y  represent  the  spatial
        dimensions.
        """
        self.input_tensor = input_tensor
        B, C_in, *spatial_in = input_tensor.shape

        # prepare stride, padding and output spatial shape
        stride, padding_tuple, output_spatial_shape = _prepare_stride_and_padding(
            input_tensor.shape, self.stride_shape, self.weights.shape
        )
        self.stride = stride
        self.padding = padding_tuple

        # Pad the input tensor with zeros
        padded_input = np.pad(input_tensor, self.padding, mode='constant')

        # Initialize the output tensor with zeros
        output_shape = (B, self.num_kernels) + tuple(output_spatial_shape)
        output_tensor = np.zeros(output_shape)

        # Create slice objects for striding to skip elements in the output to simulate stride > 1
        slices = [slice(None, None, int(s)) for s in self.stride]

        # Convolve
        for b in range(B):
            for k in range(self.num_kernels):
                kernel_k = self.weights[k]  # shape (C_in, M, N) or (C_in, M)
                # Sum the correlation results across all input channels
                if self.is_2D:
                    corr_sum = None
                    for c in range(padded_input.shape[1]):
                        corr_c = correlate(
                            padded_input[b, c],
                            kernel_k[c],
                            mode='valid'
                        )
                        if corr_sum is None:
                            corr_sum = corr_c
                        else:
                            corr_sum += corr_c
                    # corr_sum shape is (H_full, W_full) but output is subsampled
                    output_tensor[b, k] = corr_sum[tuple(slices)]
                else:
                    corr_sum = None
                    for c in range(padded_input.shape[1]):
                        corr_c = correlate(
                            padded_input[b, c],
                            kernel_k[c],
                            mode='valid'
                        )
                        if corr_sum is None:
                            corr_sum = corr_c
                        else:
                            corr_sum += corr_c
                    output_tensor[b, k] = corr_sum[tuple(slices)]

                # Add Bias to the feature map
                output_tensor[b, k] += self.bias[k]

        return output_tensor


    def backward(self, error_tensor):
        """
        error_tensor: shape (B, C_out, spatial_out...)
        require:
            updates the parameters  using the optimizer (if available) and returns the error  tensor which returns a tensor that
            servers as error tensor for the next layer.

        """
        B, C_out, *spatial_out = error_tensor.shape
        C_in = self.input_tensor.shape[1]
        input_spatial = np.array(self.input_tensor.shape[2:])

        # 1) Gradient w.r.t Bias
        # Sum the error tensor over Batch and all Spatial dimensions
        self._gradient_bias = np.sum(error_tensor, axis=(0,) + tuple(range(2, error_tensor.ndim)))

        # 2) upsample error（compensate for subsampling in forward pass)
        # forward pass used stride (downsampling), we must upsample the error
        # to match the size before striding. We fill the gaps with zeros.
        if self.is_2D:
            H_out, W_out = spatial_out[0], spatial_out[1]
            up_H = H_out * int(self.stride[0])
            up_W = W_out * int(self.stride[1])
            upsampled_error = np.zeros((B, C_out, up_H, up_W))
            upsampled_error[:, :, ::int(self.stride[0]), ::int(self.stride[1])] = error_tensor

        else:
            L_out = spatial_out[0]
            up_L = L_out * int(self.stride[0])
            upsampled_error = np.zeros((B, C_out, up_L))
            upsampled_error[:, :, ::int(self.stride[0])] = error_tensor

        # 3) compute dW (gradient w.r.t. weights) using explicit sliding-window accumulation
        # dW = Input * Error
        self._gradient_weights = np.zeros_like(self.weights)
        padded_input = np.pad(self.input_tensor, self.padding, mode='constant')
        kernel_spatial = self.weights.shape[2:]

        if self.is_2D:
            M, N = kernel_spatial
            up_H, up_W = upsampled_error.shape[2], upsampled_error.shape[3]
            for k in range(self.num_kernels):
                for c in range(C_in):
                    grad = np.zeros((M, N))
                    for b in range(B):
                        # for each upsampled error location accumulate input patch * error
                        for y in range(up_H):
                            for x in range(up_W):
                                e_val = upsampled_error[b, k, y, x]
                                if e_val == 0:
                                    continue

                                inp_patch = padded_input[b, c, y:y + M, x:x + N]

                                # Handle boundary cases where patch is smaller than kernel
                                if inp_patch.shape != (M, N):
                                    full_patch = np.zeros((M, N))
                                    h0, w0 = inp_patch.shape
                                    full_patch[:h0, :w0] = inp_patch
                                    inp_patch = full_patch
                                grad += inp_patch * e_val
                    self._gradient_weights[k, c] = grad
        else:
            M = kernel_spatial[0]
            up_L = upsampled_error.shape[2]
            for k in range(self.num_kernels):
                for c in range(C_in):
                    grad = np.zeros((M,))
                    for b in range(B):
                        for i in range(up_L):
                            e_val = upsampled_error[b, k, i]
                            if e_val == 0:
                                continue
                            inp_patch = padded_input[b, c, i:i + M]
                            if inp_patch.shape != (M,):
                                full_patch = np.zeros((M,))
                                full_patch[:inp_patch.shape[0]] = inp_patch
                                inp_patch = full_patch
                            grad += inp_patch * e_val
                    self._gradient_weights[k, c] = grad

        # 4) compute E_{n-1} (pre-layer error)
        # E_{n-1} = Error{n} * Flipped_Weights

        # Padding size = Kernel Size - 1
        pad_for_conv = [ (0,0), (0,0) ]
        for size in kernel_spatial:
            pad_for_conv.append((int(size - 1), int(size - 1)))
        padded_error_full = np.pad(upsampled_error, pad_for_conv, mode='constant')

        pre_layer_error_padded = np.zeros_like(padded_input)

        if self.is_2D:
            M, N = kernel_spatial
            P_h = padded_input.shape[2]
            P_w = padded_input.shape[3]
            for b in range(B):
                for c_in in range(C_in):
                    acc = np.zeros((P_h, P_w))
                    for k in range(self.num_kernels):
                        # flip kernel spatially for E{n-1}
                        w_kc = self.weights[k, c_in]
                        w_kc_flipped = np.flip(w_kc, axis=(0,1))

                        # convolution
                        for m in range(M):
                            for n in range(N):
                                # extract an error slice aligned with padded_input
                                err_slice = padded_error_full[b, k, m:m + P_h, n:n + P_w]
                                acc += err_slice * w_kc_flipped[m, n]
                    pre_layer_error_padded[b, c_in] = acc
            # remove forward padding to obtain E{n-1} of original input spatial size
            pad_before = [p for p in self.padding[2:]]  # exclude batch and channel entries
            y0 = pad_before[0][0]
            x0 = pad_before[1][0]
            pre_layer_error = pre_layer_error_padded[:, :, y0:y0 + input_spatial[0], x0:x0 + input_spatial[1]]
        else:
            M = kernel_spatial[0]
            P_l = padded_input.shape[2]
            for b in range(B):
                for c_in in range(C_in):
                    acc = np.zeros((P_l,))
                    for k in range(self.num_kernels):
                        w_kc = self.weights[k, c_in]
                        w_kc_flipped = np.flip(w_kc)
                        for m in range(M):
                            err_slice = padded_error_full[b, k, m:m + P_l]
                            acc += err_slice * w_kc_flipped[m]
                    pre_layer_error_padded[b, c_in] = acc
            pad_before = self.padding[2][0]
            start = pad_before
            pre_layer_error = pre_layer_error_padded[:, :, start:start + input_spatial[0]]

        # 5) update weights and bias
        if self._optimizer_weights is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return pre_layer_error
