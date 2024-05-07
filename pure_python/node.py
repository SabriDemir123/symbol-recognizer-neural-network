import random
import math

class Node:
    def __init__(self):
        self.value = 0
        self.bias = random.uniform(0, 1)
        self.incomingLinks = []
        self.outgoingLinks = []