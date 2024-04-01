from typing import List
from value import Value
import random

class Neuron:
  def __init__(self, input_size: int):
    """
    Creates a neuron with connections of size `input_size` initialized randomly
    """
    self.weights = [Value(random.uniform(-1, 1)) for _ in range (input_size)]
    self.bias = Value(random.uniform(-1, 1))
    
  def __call__(self, input_values: List[float]) -> float:
    """
    Performs the dot product of `input_values` and `self.weights`, apply
    the bias and to the result of the addition applies the activation function.
    """
    input_weight_pairs = zip(input_values, self.weights)
    result = sum((v * w for v, w in input_weight_pairs), self.bias)

    return result.tanh()