from typing import List
from neuron import Neuron

class Layer:
  def __init__(self, num_neurons: int, input_size: int):
    self.neurons = [Neuron(input_size) for _ in range(num_neurons)]
  
  def __call__(self, input_values: List[float]):
    return [n(input_values) for n in self.neurons]