from link import Link
from node import Node

class NeuralNetwork:
    def __init__(self, inputNodes, outputNodes):
        self.inputNodes = [Node() for _ in range(inputNodes)]
        self.outputNodes = [Node() for _ in range(outputNodes)]
        self.create_links()
    
    def create_links(self):
        self.totalLinks = []
        for inputNode in self.inputNodes:
            for outputNode in self.outputNodes:
                link = Link(inputNode, outputNode)
                inputNode.outgoingLinks.append(link)
                outputNode.incomingLinks.append(link)
                self.totalLinks.append(link)