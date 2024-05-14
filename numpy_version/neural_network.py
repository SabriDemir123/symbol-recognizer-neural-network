import numpy as np

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

        # Initialize weights and biases in numpy arrays
        self.weights = np.random.uniform(-1, 1, (output_nodes_size, input_nodes_size))
        self.biases = np.random.uniform(-1, 1, output_nodes_size)

        self.costs = []
        self.accuracies = []

    def sigmoid(self, x):
        """
        Compute the sigmoid function.

        Args:
            x (float): The input value.

        Returns:
            float: The output value after applying the sigmoid function.
        """
        # Sigmoid function
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, input_values):
        """
        Perform the forward propagation process.

        Args:
            input_values (list): The input values.

        Returns:
            list: Values of the output nodes.
        """
        # Calculate the weighted sum
        weighted_sum = np.dot(self.weights, input_values) + self.biases
        output_values = self.sigmoid(weighted_sum)
        return output_values


    def convert_label(self, label):
        """	
        Convert label to a numpy array with target values.

        Args:
            label (str): The label.
        
        Returns:
            np.array: The target values.
        """
        return np.array([1, 0]) if label == 'O' else np.array([0, 1])

    def calculate_mse(self, input_values, target_values):
        """
        Calculate the mean squared error.

        Args:
            input_values (list): The input values.
            target_values (str): The target values.
        
        Returns:
            float: The mean squared error.
        """
        output_values = self.forward_propagation(input_values)
        desired_values = self.convert_label(target_values)
        mse = np.mean((desired_values - output_values) ** 2) / len(output_values)
        return mse

    def back_propagation(self, input_values, target_values):
        """
        Perform the backpropagation process.

        Args:
            target_values (list): The target values.
        
        Returns:
            list: list of tuples containing the bias gradients and weight gradients.
        """
        desired_values = self.convert_label(target_values)
        output_values = self.forward_propagation(input_values)

        # Derivative of the mean squared error with respect to the output values
        derivative_mse = 2 * (output_values - desired_values)
        # Derivative of the sigmoid function
        derivative_sigmoid = output_values * (1 - output_values)
        # Calculate the bias gradients
        bias_gradients = derivative_mse * derivative_sigmoid
        # Outer product of the bias gradients and input values gives the weight gradients
        weight_gradients = np.outer(bias_gradients, input_values)

        return bias_gradients, weight_gradients

    def update_all_parameters(self, bias_gradients, weight_gradients, learning_rate):
        """
        Update all parameters of the neural network.

        Args:
            bias_gradients (list): The bias gradients.
            weight_gradients (list): The weight gradients.
            learning_rate (float): The learning rate.
        """
        self.biases -= learning_rate * bias_gradients
        self.weights -= learning_rate * weight_gradients
    

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
            flattened_input_values = np.array(input_values).flatten()
            output_values = self.forward_propagation(flattened_input_values)
            # Append prediction based on the output values
            predictions.append('O' if output_values[0] > output_values[1] else 'X')
        # Return predictions with output values
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
            # Iterate over training set
            for input_values, target_values in training_set:
                flattened_input_values = np.array(input_values).flatten()
                cost += self.calculate_mse(flattened_input_values, target_values)
                bias_gradients, weight_gradients = self.back_propagation(flattened_input_values, target_values)
                self.update_all_parameters(bias_gradients, weight_gradients, learning_rate)
            # Calculate the average cost
            cost /= len(training_set)
            self.costs.append(cost)
            # Calculate the accuracy
            accuracy = self.accuracy(training_set)
            self.accuracies.append(accuracy)
            epoch += 1
        
        print(f'Epoch: {epoch}, Cost: {cost}')