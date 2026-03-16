import numpy as np

from Layers.Base import BaseLayer


class Dropout(BaseLayer):

    def __init__(self, probability):
        """

        Implement the constructor for this layer receiving the argument probability determining
    the fraction units to keep.

        :param probability:
        """
        super().__init__()
        self.probability = probability
        self.mask = None

    @property
    def phase(self):
        return 'test' if self.testing_phase else 'train'

    @phase.setter
    def phase(self, value):
        if value == 'test':
            self.testing_phase = True
        else:
            self.testing_phase = False

    def forward(self, input_tensor):
        if not self.testing_phase: # Dropout during Training phase
            self.mask = (np.random.rand(*input_tensor.shape) < self.probability)
            return (input_tensor * self.mask) / self.probability
        else: # Don't dropout during test phase
            return input_tensor

    def backward(self, error_tensor):
        return (error_tensor * self.mask) / self.probability