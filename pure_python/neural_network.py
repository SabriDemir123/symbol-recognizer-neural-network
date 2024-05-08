from link import Link
from node import Node
import math as m

class NeuralNetwork:
    def __init__(self, input_nodes_size, output_nodes_size):
        self.input_nodes_size = input_nodes_size
        self.output_nodes_size = output_nodes_size

        self.links = []
        self.input_nodes = []
        self.output_nodes = []

        # Create all nodes
        total_nodes_size = input_nodes_size + output_nodes_size
        for i in range(total_nodes_size):
            node = Node()
            self.input_nodes.append(node) if i < input_nodes_size else self.output_nodes.append(node)

        # Create all links
        self.create_links()        

    def create_links(self):
        for input_node in self.input_nodes:
            for output_node in self.output_nodes:
                link = Link(input_node, output_node)
                input_node.outgoing_links.append(link)
                output_node.incoming_links.append(link)
                self.links.append(link)
    
    def softmax(self, values):
        total = sum([m.exp(value) for value in values])
        return [m.exp(value) / total for value in values]

    def forward_propagation(self, input_values):
        # Set input values
        index = 0
        for row in input_values:
            for value in row:
                self.input_nodes[index].value = value
                index += 1

        # Change output node values to weighted sum of incoming links
        for output_node in self.output_nodes:
            output_node.value = output_node.calculate_value()

        # Apply softmax function to output node values
        output_values = self.softmax([output_node.value for output_node in self.output_nodes])

        # Update output node values to softmax values
        for i in range(len(output_values)):
            self.output_nodes[i].value = output_values[i]
        
        return output_values