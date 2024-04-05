from typing import List
from neuron import Neuron

class Layer:
  def __init__(self, input_size: int, num_neurons: int):
    self.neurons = [Neuron(input_size) for _ in range(num_neurons)]
  
  def __call__(self, input_values: List[float]):
    outputs = [n(input_values) for n in self.neurons]
    return outputs[0] if len(outputs) == 1 else outputs