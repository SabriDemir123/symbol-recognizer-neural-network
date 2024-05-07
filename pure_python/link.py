import random

class Link:
    def __init__(self, sourceNode, destinationNode):
        self.sourceNode = sourceNode
        self.destinationNode = destinationNode
        self.weight = random.uniform(0, 1)