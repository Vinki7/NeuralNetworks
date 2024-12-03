import numpy as np
import matplotlib.pyplot as plt

"""
    Correction of the code's structure and possible refactoring done by GitHub Copilot, OpenAI ChatGPT 2024.
"""

#-------------------------------- Linear layer --------------------------------
class Linear:
    def __init__(self, num_of_inputs, num_of_outputs):# num_of_inputs: number of input neurons, num_of_outputs: number of output neurons
        self.Weight_matrix = np.random.randn(num_of_inputs, num_of_outputs) * np.sqrt(2 / num_of_inputs) # weights: input_size rows, output_size columns
        self.bias_vector = np.zeros((1, num_of_outputs))# bias: 1 row, output_size columns
        self.input_to_layer = None # input to the layer
        self.grad_W_vector = None # gradient of the loss with respect to the weights
        self.grad_b_vector = None # gradient of the loss with respect to the biases

    def forward(self, input_x):
        self.input_to_layer = input_x
        return np.dot(input_x, self.Weight_matrix) + self.bias_vector # z = Wx + b

    def backward(self, grad_output):
        self.grad_W_vector = np.dot(self.input_to_layer.T, grad_output) # dW = input * dz
        self.grad_b_vector = np.sum(grad_output, axis=0, keepdims=True) # db = dz (sum over all samples)
        return np.dot(grad_output, self.Weight_matrix.T) # dx = dz * W

#-------------------------------- Activation Functions --------------------------------
class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, neuron_result):
        self.output = 1 / (1 + np.exp(-neuron_result)) # sigmoid(x) = 1 / (1 + exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output) # sigmoid derivative is sigmoid(x) * (1 - sigmoid(x))


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, neuron_value):
        self.input = neuron_value
        return np.maximum(0, neuron_value) # ReLU(x) = max(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0 # ReLU derivative is 0 for negative values
        return grad_input


class Tanh:
    def forward(self, neuron_result):
        self.output = np.tanh(neuron_result) # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return self.output

    def backward(self, grad_output):
        # Derivative of tanh is 1 - tanh^2(x)
        return grad_output * (1 - self.output ** 2) # tanh derivative is 1 - tanh^2(x)


#-------------------------------- Loss Function --------------------------------
class MSELoss:
    def forward(self, y_predicted, y_true):
        return np.mean((y_predicted - y_true) ** 2) # MSE = 1/N * sum((y_predicted - y_true)^2)

    def backward(self, y_predicted, y_true):
        return 2 * (y_predicted - y_true) / y_true.size # dMSE = 2/N * (y_predicted - y_true)

#-------------------------------- Train --------------------------------
def train(X, y, model, loss_fn, lr, epochs=500, momentum: float = 0):
    losses = []

    # Initialize velocities for momentum
    velocities = {}
    if momentum > 0:
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                velocities[f"W_{i}"] = np.zeros_like(layer.Weight_matrix)
                velocities[f"b_{i}"] = np.zeros_like(layer.bias_vector)

    for epoch in range(1, epochs + 1):
        # Forward pass
        y_predicted = model.forward(X) # Forward pass through the network to get predictions

        # Compute loss
        loss = loss_fn.forward(y_predicted, y) # Compute the loss
        losses.append(loss)

        # Backward pass
        grad_output = loss_fn.backward(y_predicted, y) # Compute the gradient of the loss with respect to the output of the network (y_predicted)
        model.backward(grad_output) # Backward pass through the network to compute gradients of the loss with respect to the weights and biases

        # Update parameters
        model.update_params(lr, momentum, velocities) # Update the weights and biases of the network

        # Test accuracy every 50 epochs
        if epoch % 50 == 0:
            accuracy = calculate_accuracy(y_predicted, y) 
            print(f"Epoch {epoch}: Loss = {loss:.6f}, Accuracy = {accuracy:.2f}%")

    return losses

def calculate_accuracy(predictions, targets):
    """
    Calculates accuracy as the percentage of correct predictions.
    """
    rounded_predictions = np.round(predictions)  # Convert to 0 or 1
    correct = np.sum(rounded_predictions == targets)
    total = targets.size
    accuracy = correct / total * 100
    return accuracy

#-------------------------------- Neural Network --------------------------------
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x) # Forward pass through the network to get predictions
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output) # Backward pass through the network to compute gradients of the loss with respect to the weights and biases

    def update_params(self, lr, momentum:float=0, velocities=None):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                if velocities is not None and momentum > 0:
                    velocities[f'W_{i}'] = momentum * velocities[f'W_{i}'] + lr * layer.grad_W_vector # Update the velocities
                    velocities[f'b_{i}'] = momentum * velocities[f'b_{i}'] + lr * layer.grad_b_vector # Update the velocities
                    layer.Weight_matrix -= velocities[f'W_{i}'] # Update the weights
                    layer.bias_vector -= velocities[f'b_{i}'] # Update the biases
                else:
                    layer.Weight_matrix -= lr * layer.grad_W_vector # Update the weights based on the gradient and learning rate
                    layer.bias_vector -= lr * layer.grad_b_vector # Update the biases based on the gradient and learning rate

def logic_learning(model, gate_type='XOR', loss_function=MSELoss(), learning_rate=0.1, momentum:float=0, epochs=500):
    """
    Method to perform the whole training of the Neural Network

    :param model: Neural model
    :param gate_type: logical gate to train on
    :param loss_function: MSE loss
    :param learning_rate: Affection of the weights and biases
    :param momentum: Optimisation
    :param epochs: Length of learning
    :return: List of losses
    """
    # Define datasets for logic gates
    if gate_type == 'XOR':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
    elif gate_type == 'AND':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
    elif gate_type == 'OR':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
    else:
        raise ValueError("Invalid gate_type. Choose from 'XOR', 'AND', or 'OR'.")

    # Train the model
    losses = train(X, y, model, loss_function, lr=learning_rate, momentum=momentum, epochs=epochs)
    y_predicted = model.forward(X)
    accuracy = calculate_accuracy(y_predicted, y)

    print(f"\nFinal Loss ({gate_type}): {losses[-1]:.6f}")
    print(f"Final Accuracy ({gate_type}): {accuracy:.2f}%\n")
    return losses

def plot_losses_subplots(losses_dicts, titles, epochs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create 1 row, 3 columns
    for ax, losses_dict, title in zip(axes, losses_dicts, titles):
        for label, losses in losses_dict.items():
            ax.plot(range(epochs), losses, label=label)
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

def run(epochs):
    # np.random.seed(7)
    # ---------------------------------- Helper functions ----------------------------------
    def create_model_sig():
        return NeuralNetwork([
            Linear(2, 4),
            Sigmoid(),
            # Linear(4, 4),
            # Sigmoid(),
            Linear(4, 1),
            Sigmoid()
        ])

    def create_model_relu():
        return NeuralNetwork([
            Linear(2, 4),
            ReLU(),
            # Linear(4, 4),
            # ReLU(),
            Linear(4, 1),
            ReLU()
        ])

    def create_model_tanh():
        return NeuralNetwork([
            Linear(2, 4),
            Tanh(),
            # Linear(4, 4),
            # Tanh(),
            Linear(4, 1),
            Tanh()
        ])

    #---------------------------------- Sigmoid ----------------------------------
    xor_losses_sig = logic_learning(create_model_sig(), gate_type='XOR', learning_rate=0.04, momentum=0, epochs=epochs)
    xor_losses_sig_mom = logic_learning(create_model_sig(), gate_type='XOR', learning_rate=0.06, momentum=0.9, epochs=epochs) # should work like that

    and_losses_sig = logic_learning(create_model_sig(), gate_type='AND', learning_rate=0.05, momentum=0, epochs=epochs)# this is done, ready for documenting
    and_losses_sig_mom = logic_learning(create_model_sig(), gate_type='AND', learning_rate=0.05, momentum=0.9, epochs=epochs)# this is done, ready for documenting

    or_losses_sig = logic_learning(create_model_sig(), gate_type='OR', learning_rate=0.05, momentum=0, epochs=epochs)# this is done, ready for documenting
    or_losses_sig_mom = logic_learning(create_model_sig(), gate_type='OR', learning_rate=0.05, momentum=0.9, epochs=epochs)# this is done, ready for documenting


    #---------------------------------- Relu ----------------------------------
    xor_losses_relu = logic_learning(create_model_relu(), gate_type='XOR', learning_rate=0.05, momentum=0, epochs=epochs)
    xor_losses_relu_mom = logic_learning(create_model_relu(), gate_type='XOR', learning_rate=0.05, momentum=0.9,
                                        epochs=epochs)

    and_losses_relu = logic_learning(create_model_relu(), gate_type='AND', learning_rate=0.05, momentum=0, epochs=epochs)
    and_losses_relu_mom = logic_learning(create_model_relu(), gate_type='AND', learning_rate=0.02, momentum=0.9,
                                        epochs=epochs)

    or_losses_relu = logic_learning(create_model_relu(), gate_type='OR', learning_rate=0.05, momentum=0, epochs=epochs)
    or_losses_relu_mom = logic_learning(create_model_relu(), gate_type='OR', learning_rate=0.02, momentum=0.9, epochs=epochs)

    #---------------------------------- Tanh ----------------------------------
    xor_losses_tanh = logic_learning(create_model_tanh(), gate_type='XOR', learning_rate=0.08, momentum=0, epochs=epochs) # this is donre, ready for documentation
    xor_losses_tanh_mom = logic_learning(create_model_tanh(), gate_type='XOR', learning_rate=0.08, momentum=0.9,
                                         epochs=epochs) # this is donre, ready for documentation

    and_losses_tanh = logic_learning(create_model_tanh(), gate_type='AND', learning_rate=0.08, momentum=0, epochs=epochs) # this is donre, ready for documentation
    and_losses_tanh_mom = logic_learning(create_model_tanh(), gate_type='AND', learning_rate=0.08, momentum=0.9,
                                         epochs=epochs) # this is donre, ready for documentation

    or_losses_tanh = logic_learning(create_model_tanh(), gate_type='OR', learning_rate=0.08, momentum=0, epochs=epochs) # this is donre, ready for documentation
    or_losses_tanh_mom = logic_learning(create_model_tanh(), gate_type='OR', learning_rate=0.08, momentum=0.9,
                                        epochs=epochs) # this is donre, ready for documentation

    # ---------------------------------- Plotting ----------------------------------
    plot_losses_subplots(
        [
            {
                "Sigmoid": xor_losses_sig,
                "Sigmoid + Momentum": xor_losses_sig_mom,
                "ReLU": xor_losses_relu,
                "ReLU + Momentum": xor_losses_relu_mom,
                "Tanh": xor_losses_tanh,
                "Tanh + Momentum": xor_losses_tanh_mom,
            },
            {
                "Sigmoid": and_losses_sig,
                "Sigmoid + Momentum": and_losses_sig_mom,
                "ReLU": and_losses_relu,
                "ReLU + Momentum": and_losses_relu_mom,
                "Tanh": and_losses_tanh,
                "Tanh + Momentum": and_losses_tanh_mom,
            },
            {
                "Sigmoid": or_losses_sig,
                "Sigmoid + Momentum": or_losses_sig_mom,
                "ReLU": or_losses_relu,
                "ReLU + Momentum": or_losses_relu_mom,
                "Tanh": or_losses_tanh,
                "Tanh + Momentum": or_losses_tanh_mom,
            }
        ],
        ["XOR Training Losses", "AND Training Losses", "OR Training Losses"],
        epochs
    )

if __name__ == "__main__":
    run(epochs=1000)