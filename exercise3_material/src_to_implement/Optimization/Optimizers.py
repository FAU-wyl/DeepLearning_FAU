from abc import abstractmethod

import numpy as np

class Optimizer:
    """
    Create  a  base-class  Optimizer  for  optimizers  in  the  “Optimizers.py”  file.   Make  all
    optimizers inherit from this “base-optimizer”.
    The class Optimizer should have a method add  regularizer(regularizer) and a member
    storing the regularizer.
    """

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        """
        The class Optimizer should have a method add  regularizer(regularizer) and a member
        storing the regularizer.

        """
        self.regularizer = regularizer

    @abstractmethod
    def calculate_update(self, weight_tensor, gradient_tensor):
        pass

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = np.float64(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Return:
             The updated weights after applying the gradient descent update.
        """

        """ 
        Refactor  the  optimizers  to  apply  the  new  regularizer  if  it  is  set  using  the  calcu late gradient(weights) method.
        """
        updated_weights = weight_tensor
        if self.regularizer:
            # Shrinkage the weight
            updated_weights = updated_weights - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)


        #  SGD update
        updated_weights = updated_weights - self.learning_rate * gradient_tensor
        return updated_weights


class SgdWithMomentum(Optimizer):
    """
    带有动量的随机梯度下降 (SGD with Momentum) 优化器。
    """
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Refactor  the  optimizers  to  apply  the  new  regularizer  if  it  is  set  using  the  calcu-
        # late  gradient(weights) method.
        updated_weight_tensor = weight_tensor
        if self.regularizer:
            # Shrinkage the weight
            updated_weight_tensor = updated_weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        # 1. Initialize velocity
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)

        # 2. calculate v_k
        # v_k = μ * v_{k-1} - η * ∇ J(w)
        self.velocity = (self.momentum_rate * self.velocity) - (self.learning_rate * gradient_tensor)

        # 3. update w
        # w_{k+1} = w_k + v_k
        updated_weight_tensor = updated_weight_tensor + self.velocity

        return updated_weight_tensor


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
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

        update_step = self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        new_weights = weight_tensor - update_step

        if self.regularizer: # Shrinkage the weight
            # w_new = w_old - Adam_step - lr * reg_grad
            new_weights -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return new_weights