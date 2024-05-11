import random

class Node:
    def __init__(self):
        """
        Initialize a node with a random bias and no incoming or outgoing links.
        """
        self.value = 0.0
        self.bias = random.uniform(-1, 1)
        self.incoming_links = []
        self.outgoing_links = []

    def calculate_value(self):
        """
        Calculate the value of the node with the weighted sum of its incoming links and bias.
        """
        weighted_sum = sum([link.source_node.value * link.weight for link in self.incoming_links]) + self.bias
        return weighted_sum