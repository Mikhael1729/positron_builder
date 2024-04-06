from layer import Layer
from value import Value
from typing import List

class MLP:
  def __init__(self, input_size: int, layer_sizes: List[int]):
    sizes = [input_size] + layer_sizes

    self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(layer_sizes))]

  def __call__(self, inputs: List[float]) -> List[Value] | Value:
    for layer in self.layers:
      inputs = layer(inputs)
    
    return inputs
  
  def parameters(self) -> List[Value]:
    """
    Returns the list containing all the parameteres inside
    each layer of the network
    """
    return [p for l in self.layers for p in l.parameters()]
    
if __name__ == "__main__":
  positron = MLP(3, [4, 4, 1])
  thought = [2.0, 3.0, -1.0]

  positron(thought)

  print(len(positron.layers))