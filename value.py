from __future__ import annotations
from typing import List, Set, Tuple, TypeAlias
import math

class Value:
  """
  Value is encodes the data and computation information of how the given result or
  value was obtained.
  """
  def __init__(self, data: int | float, label='', parents: Tuple[Value]=(), _operator=''):
    self.data = data
    """
    An scalar value encoding the information in the given value.
    """

    self.label = label
    """
    A text that can label a given value (mainly used for debugging)
    """

    self.gradient = 0.0 # This is computed in the backwards pass
    """
    This value is computed when _backward gets called. It stores the gradient for
    the given value.
    """

    self._operator = _operator
    """
    Indicates the operator used in the operation that resulted in the given value.
    If the given value is not the result of an operation, this value is empty.
    """

    self._parents: Set[Value] = set(parents)
    """
    Contains the parent operands of the given value if any
    """

    self._backward = lambda: None
    """
    A function that sets the value for the gradients of the Value operands involved
    in a given operation (computational node like +, *, etc.).
    """

  def __repr__(self) -> str:
    """
    Overrides the string parsing operation for Value
    """
    labels = [child.label for child in self._parents]
    if len(labels) == 2:
      parents = (labels[0], labels[1])
    else:
      parents = ()

    return f'Value(data={self.data}, label="{self.label}", parents={parents})'

  def __add__(self, other: ValueType) -> Value:
    """
    Overrides the addition operation for value (self + other)
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
    Overrides the negation operation for value (-self)
    """
    return self * -1

  def __sub__(self, other: ValueType) -> Value:
    """
    Overrides the subtraction operation for Value (self - other)
    """
    return self + (-other)

  def __mul__(self, other: ValueType) -> Value:
    """
    Overrides the multiplication operation for Value (self * other)
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
    Overrides the reverse multiplication (when other is other type). In short,
    it allows the operation other * self to behave self * other
    """
    return self * other
  
  def __pow__(self, other: int | float) -> Value:
    """
    Overrides the exponent operation for Value. It only supports numeric values as
    exponents
    """
    assert isinstance(other, (int, float)), "only supporting int/float numbers for now"

    result = Value(
      data = self.data**other,
      parents = (self, ), # Other is not included, because the exponent is used a as a constant which do not affect differentiation
      _operator = f"**{other}"
    )

    def backward():
      self.gradient += (other * self.data **(other-1)) * result.gradient

    result._backward = backward

    return result

  def __truediv__(self, other: ValueType) -> Value:
    """
    Overrides the division operation of Value in terms of factors

    derivation:\n
    result = self / other\n
    result = self * (1 / b)\n
    result = self * (b**-1)\n
    """
    return self * other**-1

  def exp(self) -> Value:
    """
    Operator that applies the exponential function with Value (`e^(self.data)`)
    """
    result = Value(
      data = math.exp(self.data),
      parents = (self, ),
      _operator = 'exp'
    )

    def backward():
      self.gradient += result.data * result.gradient

    result._backward = backward

    return result

  def tanh(self: Value) -> Value:
    """
    Operator that applies the tanh function to Value (`tanh(self.data)`)
    """
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
    """
    Computes the gradient for the given value
    """
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

    for v in value._parents:
      Value._get_parents_in_reverse_computational_order_callback(v, nodes, visited)

    nodes.append(value)

ValueType: TypeAlias = Value | int | float

def parse(other: ValueType) -> Value:
  return other if isinstance(other, Value) else Value(other)

def label(value: Value, label: str) -> Value:
  """
  A helper function to allow you to easily (in a single line) assign a label to a
  given value
  """
  value.label = label
  return value
