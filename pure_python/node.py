import random
import math

class Node:
    def __init__(self):
        self.value = 0.0
        self.bias = random.uniform(-1, 1)
        self.incoming_links = []
        self.outgoing_links = []

    def calculate_value(self):
        weighted_sum = sum([link.source_node.value * link.weight for link in self.incoming_links]) + self.bias
        return weighted_sum