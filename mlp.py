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

  def predict(self, batch: List[float]) -> List[Value]:
    return [self(sample) for sample in batch]
  
  def train(self, batch: List[float], expected_values: List[float], iterations: int, step_size: float, debug=True):
    for i in range(iterations):
      predicted_values = [self(sample) for sample in batch]

      # Compute loss of results
      loss: Value = sum((p - y)**2 for y, p in zip(expected_values, predicted_values))

      if debug:
        print(f"{i + 1}: {loss.data}")

      # Compute the gradients of the network with respect to the loss
      self.reset_parameters_gradients()
      loss.backward()

      # Tune the data of the parameters according to the gradient information
      for parameter in self.parameters():
        parameter.data += -step_size * parameter.gradient # We are updating it to go at the contrary direction of the loss (that's the reason of the -)

  def reset_parameters_gradients(self):
    """
    Resets the gradients of each parameter in the network. It is useful
    for training, to avoid affecting new calculations with past data
    """
    for parameter in self.parameters():
      parameter.gradient = 0.0
    
if __name__ == "__main__":
  positron = MLP(3, [4, 4, 1])
  thought = [2.0, 3.0, -1.0]

  positron(thought)

  print(len(positron.layers))