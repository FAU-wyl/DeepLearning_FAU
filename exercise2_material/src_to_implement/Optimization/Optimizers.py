import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = np.float64(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Return:
             The updated weights after applying the gradient descent update.
        """
        # SGD Function: w‘ = w - learning_rate * gradient
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights


class SgdWithMomentum:
    """
    带有动量的随机梯度下降 (SGD with Momentum) 优化器。
    """

    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        # 1. Initialize velocity
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)

        # 2. calculate v_k
        # v_k = μ * v_{k-1} - η * ∇ J(w)
        self.velocity = (self.momentum_rate * self.velocity) - (self.learning_rate * gradient_tensor)

        # 3. update w
        # w_{k+1} = w_k + v_k
        updated_weight_tensor = weight_tensor + self.velocity

        return updated_weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.k = 0  # exponent

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Initialize
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
            self.v = np.zeros_like(weight_tensor)

        self.k += 1

        # update Momentum
        # m_k = mu * m_{k-1} + (1 - mu) * g
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor

        # 4. update Velocity
        # v_k = rho * v_{k-1} + (1 - rho) * g^2
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)

        # 5. Bias Correction
        # m_hat = m_t / (1 - mu^t)
        # v_hat = v_t / (1 - rho^t)
        m_hat = self.m / (1 - self.mu ** self.k)
        v_hat = self.v / (1 - self.rho ** self.k)

        # 6. update w
        # w_{k+1} = w_k - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        epsilon = 1e-8
        updated_weight = weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        return updated_weight