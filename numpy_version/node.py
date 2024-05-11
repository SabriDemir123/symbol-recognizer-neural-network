import numpy as np
    
class Node:
    def __init__(self):
        self.value = 0.0
        self.bias = np.random.uniform(-1, 1)
        self.incoming_links = []
        self.outgoing_links = []

    def calculate_value(self):
        # Calculate the weighted sum using matrix multiplication
        weights = np.array([link.weight for link in self.incoming_links])
        values = np.array([link.source_node.value for link in self.incoming_links])
        weighted_sum = np.dot(values, weights) + self.bias
        return weighted_sum