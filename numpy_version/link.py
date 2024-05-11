import numpy as np

class Link:
    def __init__(self, source_node, destination_node):
        self.source_node = source_node
        self.destination_node = destination_node
        self.weight = np.random.uniform(-1, 1)