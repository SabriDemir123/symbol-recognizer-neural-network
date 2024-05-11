from link import Link
from node import Node
import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes_size, output_nodes_size):
        self.input_nodes_size = input_nodes_size
        self.output_nodes_size = output_nodes_size

        self.links = []
        self.input_nodes = []
        self.output_nodes = []

        self.costs = []
        self.accuracies = []

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
                # Create a link between each input node and each output node
                link = Link(input_node, output_node)
                # Add the link to the lists of outgoing links of the input node and incoming links of the output node
                input_node.outgoing_links.append(link)
                output_node.incoming_links.append(link)
                self.links.append(link)

    def sigmoid(self, x):
        # Sigmoid function
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, input_values):
        input_values = np.array(input_values).flatten()
        
        # Set input values
        for i, node in enumerate(self.input_nodes):
            node.value = input_values[i]

        output_values = []
        for output_node in self.output_nodes:
            output_values.append(output_node.calculate_value())

        # Apply the sigmoid function to the output values
        for i, node in enumerate(self.output_nodes):
            node.value = self.sigmoid(output_values[i])

        return [node.value for node in self.output_nodes]


    def convert_label(self, label):
        return np.array([1, 0]) if label == 'O' else np.array([0, 1])

    def calculate_mse(self, input_values, target_values):
        output_values = self.forward_propagation(input_values)
        desired_values = self.convert_label(target_values)
        # Mean squared error loss function
        return np.mean((desired_values - output_values) ** 2) / len(output_values)

    def back_propagation(self, target_values):
        desired_values = self.convert_label(target_values)

        # Initialize gradient matrices
        bias_gradients = np.zeros(len(self.output_nodes))
        weight_gradients = np.zeros((len(self.output_nodes), len(self.input_nodes)))

        for i, node in enumerate(self.output_nodes):
            # Calculate the derivative of the mean squared error loss function
            derivative_mse = 2 * (node.value - desired_values[i])
            # Calculate the derivative of the sigmoid function
            derivative_sigmoid = node.value * (1 - node.value)
            # Compute bias gradients
            bias_gradients[i] = derivative_mse * derivative_sigmoid
            # Collect values of the source nodes of incoming links into a NumPy array
            source_node_values = np.array([link.source_node.value for link in node.incoming_links])
            # Calculate weight gradients using element-wise multiplication
            weight_gradients[i] = bias_gradients[i] * source_node_values

        return bias_gradients, weight_gradients

    def update_all_parameters(self, bias_gradients, weight_gradients, learning_rate):
        # Update biases
        for i, node in enumerate(self.output_nodes):
            node.bias -= learning_rate * bias_gradients[i]

        # Update weights
        for i, node in enumerate(self.output_nodes):
            for j, link in enumerate(node.incoming_links):
                link.weight -= learning_rate * weight_gradients[i][j]
    

    def predict(self, test_set):
        predictions = []
        for input_values, target_values in test_set:
            output_values = self.forward_propagation(input_values)
            # Append prediction based on the output values
            predictions.append('O' if output_values[0] > output_values[1] else 'X')
        # Return predictions with output values
        return predictions

    def accuracy(self, training_set):
        # Make predictions on the training set using the predict method
        predictions = self.predict(training_set)
        # Extract the true labels from the training set
        true_labels = np.array([label for _, label in training_set])
        # Check how many predictions match the true labels
        correct = np.sum(np.array(predictions) == true_labels)

        return correct / len(training_set) * 100

    def train(self, training_set, max_cost, max_epoch, learning_rate):
        # Ensure loop runs at least once
        cost = max_cost + 1
        epoch = 0
        while epoch < max_epoch and cost > max_cost:
            cost = 0
            # Iterate over training set
            for input_values, target_values in training_set:
                cost += self.calculate_mse(input_values, target_values)
                bias_gradients, weight_gradients = self.back_propagation(target_values)
                self.update_all_parameters(bias_gradients, weight_gradients, learning_rate)
            # Calculate the average cost
            cost /= len(training_set)
            self.costs.append(cost)
            # Calculate the accuracy
            accuracy = self.accuracy(training_set)
            self.accuracies.append(accuracy)
            epoch += 1
        
        print(f'Epoch: {epoch}, Cost: {cost}')