import numpy as np

class Link:
    def __init__(self, source_node, destination_node):
        """
        Initialize a link between two nodes with a random weight.

        Args:
            source_node (Node): The source node of the link.
            destination_node (Node): The destination node of the link.
        """
        self.source_node = source_node
        self.destination_node = destination_node
        self.weight = np.random.uniform(-1, 1)