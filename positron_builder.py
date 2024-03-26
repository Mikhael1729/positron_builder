import math

class Value:
  """
  Value encodes the data and computation information of how the given result or
  value was obtained.

  Attributes:
    - _backward (Callable[None, None]): A lambda function that  sets the value
      for the gradients of the two values involved in a given operation
      (computational node like +, *, etc.)
                          
  """
  def __init__(self, data, label='', parents=(), _operator=''):
    self.data = data
    self._prev = set(parents)
    self.gradient = 0.0 # This is computed in the backwards pass
    self._backward = lambda: None
    self._operator = _operator
    self.label = label

  def __repr__(self) -> str:
    labels = [child.label for child in self._prev]
    if len(labels) == 2:
      parents = (labels[0], labels[1])
    else:
      parents = ()

    return f'Value(data={self.data}, label="{self.label}", parents={parents})'

  def __add__(self, other) -> Value:
    result = Value(
      data = self.data + other.data,
      parents = (self, other),
      _operator = '+'
    )

    def backward():
      self.grad = 1.0 * result.gradient
      result.grad = 1.0 * result.gradient

    result._backward = backward

    return result

  def __mul__(self, other) -> Value:
    result = Value(
      data = self.data * other.data,
      parents = (self, other),
      _operator = '*'
    )

    def backward():
      self.grad = other.data * result.grad
      other.grad = self.data * result.grad

    result._backward = backward

    return result

  def tanh(self) -> Value:
    t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)

    result = Value(
      data = t,
      parents = (self, ),
      _operator = 'tanh'
    )

    def backward():
      self.grad = (1 - t**2) * result.grad

    result._backward = backward

    return result


def label(value: Value, label: str) -> Value:
    value.label = label
    return value
