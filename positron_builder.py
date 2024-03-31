from __future__ import annotations
from typing import List, Set, Union, TypeAlias
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
  def __init__(self, data: int | float, label='', parents=(), _operator=''):
    self.data = data
    self._prev = set(parents)
    self.gradient = 0.0 # This is computed in the backwards pass
    self._backward = lambda: None
    self._operator = _operator
    self.label = label

  def __repr__(self) -> str:
    """
    Overrides the parsing to string of Value
    """
    labels = [child.label for child in self._prev]
    if len(labels) == 2:
      parents = (labels[0], labels[1])
    else:
      parents = ()

    return f'Value(data={self.data}, label="{self.label}", parents={parents})'

  def __add__(self, other: ValueType) -> Value:
    """
    Overrides the operation self + other
    """
    other = parse(other)

    result = Value(
      data = self.data + other.data,
      parents = (self, other),
      _operator = '+'
    )

    def backward():
      self.gradient += 1.0 * result.gradient
      other.gradient += 1.0 * result.gradient

    result._backward = backward

    return result

  def __neg__(self) -> Value:
    """
    Overrides the operation -self
    """
    return self * -1

  def __sub__(self, other: ValueType) -> Value:
    """
    Overrides self - other
    """
    return self + (-other)

  def __mul__(self, other: ValueType) -> Value:
    """
    Overrides the operation self * other
    """
    other = parse(other)

    result = Value(
      data = self.data * other.data,
      parents = (self, other),
      _operator = '*'
    )

    def backward():
      self.gradient += other.data * result.gradient
      other.gradient += self.data * result.gradient

    result._backward = backward

    return result

  def __rmul__(self, other: ValueType) -> Value:
    """
    Overrides the operation other * self to behave as self * other
    """
    return self * other
  
  def __pow__(self, other: int | float) -> Value:
    assert isinstance(other, (int, float)), "only supporting int/float numbers for now"

    result = Value(
      data = self.data**other,
      parents = (self, ), # Other is not included, because the exponent is used a as a constant which do not affect differentiation
      _operator = f"**{other}"
    )

    def backward():
      self.gradient += (other * self.data **(other-1)) * result.gradient

    self._backward = backward

    return result

  def __truediv__(self, other: ValueType) -> Value:
    """
    Override division operation in terms of factors

    result = self / other
    result = self * (1 / b)
    result = self * (b**-1)
    """
    return self * other**-1

  def exp(self) -> Value:
    result = Value(
      data = math.exp(self.data),
      parents = (self, ),
      _operator = 'exp'
    )

    def backward():
      self.gradient += self.data * result.gradient

    result._backward = backward

    return result

  def tanh(self: Value) -> Value:
    t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)

    result = Value(
      data = t,
      parents = (self, ),
      _operator = 'tanh'
    )

    def backward():
      self.gradient += (1 - t**2) * result.gradient

    result._backward = backward

    return result

  def backward(self):
    self.gradient = 1.0

    nodes = self._get_parents_in_reverse_computational_order()

    for node in nodes:
      node._backward()

  def _get_parents_in_reverse_computational_order(self):
    nodes = []
    visited = set()

    Value._get_parents_in_reverse_computational_order_callback(self, nodes, visited)

    return reversed(nodes)


  def _get_parents_in_reverse_computational_order_callback(value: Value, nodes: List[Value], visited: Set[Value]) -> List[Value]:
    if value in visited:
      return

    visited.add(value)

    for v in value._prev:
      Value._get_parents_in_reverse_computational_order_callback(v, nodes, visited)

    nodes.append(value)

ValueType: TypeAlias = Value | int | float

def parse(other: ValueType) -> Value:
  return other if isinstance(other, Value) else Value(other)

def label(value: Value, label: str) -> Value:
    value.label = label
    return value
