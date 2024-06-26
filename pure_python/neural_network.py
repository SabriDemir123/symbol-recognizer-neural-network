from link import Link
from node import Node
import math as m

class NeuralNetwork:
    def __init__(self, input_nodes_size, output_nodes_size):
        """
        Initialize the neural network with the number of input nodes and output nodes.

        Args:
            input_nodes_size (int): The number of input nodes.
            output_nodes_size (int): The number of output nodes.
        """
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
        """
        Create links between input nodes and output nodes.
        """
        for input_node in self.input_nodes:
            for output_node in self.output_nodes:
                link = Link(input_node, output_node)
                input_node.outgoing_links.append(link)
                output_node.incoming_links.append(link)
                self.links.append(link)
    
    def softmax(self, values):
        """
        Compute the softmax function.

        Args:
            values (list): The input values.
        
        Returns:
            list: The output values after applying the softmax function.
        """
        total = sum([m.exp(value) for value in values])
        return [m.exp(value) / total for value in values]

    def convert_label(self, label):
        """
        Convert the label to a list of desired values.

        Args:
            label (str): The label.

        Returns:
            list: The list of desired values.
        """
        return [1, 0] if label == 'O' else [0, 1]

    def forward_propagation(self, input_values):
        """
        Perform the forward propagation process.

        Args:
            input_values (list): The input values.

        Returns:
            list: Values of the output nodes.
        """
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
    
    def calculate_mse(self, input_values, target_values):
        """
        Calculate the mean squared error.

        Args:
            input_values (list): The input values.
            target_values (str): The target values.
        
        Returns:
            float: The mean squared error.
        """
        # Calculate output with forward propagation
        output_values = self.forward_propagation(input_values)

        # Get desired values
        desired_values = self.convert_label(target_values)

        # Calculate mean squared error
        return sum([m.pow((desired_values[i] - output_values[i]), 2) for i in range(len(output_values))]) / len(output_values)
    
    def back_propagation(self, target_values):
        """
        Perform the backpropagation process.

        Args:
            target_values (list): The target values.
        
        Returns:
            list: list of tuples containing the bias gradients and weight gradients.
        """
        desired_values = self.convert_label(target_values)
    
        # Initialize the gradient as an empty list
        bias_gradiants = []
        weight_gradients = []
    
        for i, node in enumerate(self.output_nodes):
            # Calculate the derivative of the mean squared error loss function
            derivative_mse = 2 * (node.value - desired_values[i])
            # Calculate the derivative of the softmax function
            derivative_soft_max = node.value * (1 - node.value)
            # Append the gradient of the bias to the list
            bias_gradiants.append((node, derivative_mse * derivative_soft_max))
            # Iterate over each incoming link to the current output node
            for link in node.incoming_links:
                # Calculate the derivative of the output value with respect to the link weight
                derivative_output_value = link.source_node.value
                # Append the gradient of the link weight to the list
                weight_gradients.append((link, derivative_mse * derivative_soft_max * derivative_output_value))
                
        return bias_gradiants, weight_gradients
    
    def update_all_parameters(self, bias_gradients, weight_gradients, learning_rate):
        """
        Update all parameters of the neural network.

        Args:
            bias_gradients (list): The bias gradients.
            weight_gradients (list): The weight gradients.
            learning_rate (float): The learning rate.
        """
        for node, bias_gradient in bias_gradients:
            node.bias -= learning_rate * bias_gradient

        for link, weight_gradient in weight_gradients:
            link.weight -= learning_rate * weight_gradient
    
    def predict(self, test_set):
        """
        Make predictions on a given test set.

        Args:
            test_set (list): The test set.

        Returns:
            list: The predictions.
        """
        predictions = []
        for input_values, target_values in test_set:
            # Calculate output with forward propagation
            output_values = self.forward_propagation(input_values)
            # Choose label with highest probability
            predictions.append('O' if output_values[0] > output_values[1] else 'X')
        return predictions

    def accuracy(self, test_set):
        """
        Calculate the accuracy of the neural network.

        Args:
            training_set (list): The test set.
        
        Returns:
            float: The accuracy of the neural network.
        """
        correct = 0
        # Get predictions
        predictions = self.predict(test_set)
        # Compare predictions with target values
        for i in range(len(predictions)):
            if predictions[i] == test_set[i][1]:
                correct += 1
        # Calculate accuracy
        accuracy = correct / len(test_set) * 100
        return accuracy

    def train(self, training_set, max_cost, max_epoch, learning_rate):
        """
        Train the neural network with a given training set.

        Args:
            training_set (list): The training set.
            max_cost (float): The maximum cost.
            max_epoch (int): The maximum number of epochs.
            learning_rate (float): The learning rate.
        """
        # Ensure loop runs at least once
        cost = max_cost + 1
        epoch = 0
        while epoch < max_epoch and cost > max_cost:
            cost = 0
            # Iterate over each training example
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