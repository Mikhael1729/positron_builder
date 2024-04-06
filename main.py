from mlp import MLP
from value import Value, label
from graphviz import Digraph
from typing import Tuple, Set

import os

def main():
  """
  Here I play with the different capabilities of the core libraries
  """
  # e1_show_basics_of_value_operations()
  # e2_create_a_raw_aritficial_neuron()
  # e3_play_with_addition_and_substraction()
  # e4_create_a_raw_aritficial_neuron()
  # e5_mlp_neural_network_architecture()
  # e6_testing_parameters()
  # e7_predict_results()
  e8_one_iteration_of_optimization()

def e8_one_iteration_of_optimization():
  # Input data
  X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
  ]

  # Expected outputs
  Y = [1.0, -1.0, -1.0, 1.0]

  # Building network
  positron = MLP(3, [4, 4, 1])

  # --- From here repeat until you know you hit the target
  # Predicted results
  P = [positron(x) for x in X]

  # Compute loss of results
  loss: Value = sum((p - y)**2 for y, p in zip(Y, P))

  # Compute the gradients of the network with respect to the loss
  loss.backward()

  # Tune the data of the parameters according to the gradient information
  for parameter in positron.parameters():
    parameter.data += -0.01 * parameter.gradient # We are updating it to go at the contrary direction of the loss (that's the reason of the -)
  # --- here

  print(P)

def e7_predict_results():
  # Input data
  X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
  ]

  # Expected outputs
  Y = [1.0, -1.0, -1.0, 1.0]

  # Building network
  positron = MLP(3, [4, 4, 1])

  # Predicted results
  P = [positron(x) for x in X]

  for value in P:
    print(value)

def e6_testing_parameters():
  positron = MLP(3, [4, 4, 1])
  for (i, parameter) in enumerate(positron.parameters()):
    print(f"{i + 1}: {parameter}")


def e5_mlp_neural_network_architecture():
  # Creates a neural network with 3 layers:
  # 
  # 1. Hidden layer A containing 4 neurons
  # 2. Hidden layer B containing 4 neurons
  # 3. Output layer containing 1 neuron
  #
  # The "input layer" is indicated apart in the first parameter
  positron = MLP(3, [4, 4, 1])

  # These are the inputs applied to the first layer (A) of the network
  inputs = [2.0, 3.0, -1.0]

  # It perform a forward pass in the network
  thought = positron(inputs)
  print(thought)

  # Visualize the network
  graph = generate_computation_graph_from_node(thought, 'LR')
  draw_computation_graph(graph)


def e4_create_a_raw_aritficial_neuron():
  # Inputs x1, x2
  x1 = Value(2.0, 'x1')
  x2 = Value(0.0, 'x2')

  # Weights
  w1 = Value(-3.0, 'w1')
  w2 = Value(1.0, 'w2')

  # Bias of the neuron
  b = Value(6.8813735870195432, 'b') # The reason of the large number is to allow nice values
  
  # Forward propagation
  y1 = label(x1 * w1, 'y1')
  y2 = label(x2 * w2, 'y2')
  y = label(y1 + y2, 'y')

  z = label(y + b, 'z')

  # Compute tanh
  e = (2 * z).exp()
  a = label((e - 1) / (e + 1), 'a')

  a.backward()

  # Visualize computational graph
  graph = generate_computation_graph_from_node(a, 'LR')
  draw_computation_graph(graph)

def e3_play_with_addition_and_substraction():
  a = Value(2.0)
  b = Value(3.0)
  print(b - a)

def e2_create_a_raw_aritficial_neuron():
  # Inputs x1, x2
  x1 = Value(2.0, 'x1')
  x2 = Value(0.0, 'x2')

  # Weights
  w1 = Value(-3.0, 'w1')
  w2 = Value(1.0, 'w2')

  # Bias of the neuron
  b = Value(6.8813735870195432, 'b') # The reason of the large number is to allow nice values
  
  # Forward propagation
  y1 = label(x1 * w1, 'z1')
  y2 = label(x2 * w2, 'z2')
  y = label(y1 + y2, 'y')

  z = label(y + b, 'z')

  a = label(z.tanh(), 'a') # Activation function

  a.backward()

  # Visualize computational graph
  graph = generate_computation_graph_from_node(a, 'TB')
  draw_computation_graph(graph)
  

def e1_show_basics_of_value_operations():
  # Form computation
  a = Value(2.0, 'a')
  b = Value(-3.0, 'b')
  c = Value(10.0, 'c')
  e = label(a * b, 'e')
  d = label(e + c, 'd')
  f = Value(-2.0, 'f')
  L = label(d*f, 'L')

  # Perform backward pass
  L.backward()

  # Visualize computation (forward and backward pass)
  graph = generate_computation_graph_from_node(L, "TB")
  draw_computation_graph(graph)


def trace(root: Value):
  """
  Returns the nodes and edges of neighbor values used to form a graph
  """
  nodes = set()
  edges = set()
  
  def build(value: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    if value not in nodes:
      nodes.add(value)

      for child in value._parents:
        edges.add((child, value))
        build(child)

  build(root)
  
  return nodes, edges

def generate_computation_graph_from_node(root, drawing_direction='TB'):
  """
  Draw the computation graph from the given node
  """
  dot = Digraph(format='svg', graph_attr={'rankdir': drawing_direction}) # TB top to bottom

  nodes, edges = trace(root)

  for node in nodes:
    uid = str(id(node))

    dot.node(
      name = uid,
      label = "{ %s | d: %.4f | g: %.4f }" % (node.label, node.data, node.gradient),
      shape = 'record',
    )

    if node._operator:
      dot.node(name = uid + node._operator, label = node._operator)
      dot.edge(uid + node._operator, uid)

  for neighbor1, neighbor2, in edges:
    dot.edge(str(id(neighbor1)), str(id(neighbor2)) + neighbor2._operator)

  return dot

def draw_computation_graph(dot, output_filename="./visualization/computation_graph", view=True):
  unique_filename = output_filename
  counter = 1

  while os.path.exists(f'{unique_filename}.svg'):
      unique_filename = f"{output_filename}_{counter}"
      counter += 1

  dot.render(unique_filename, view=True)

if __name__ == "__main__":
  main()
