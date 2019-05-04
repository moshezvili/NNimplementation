import numpy as np


class DNN:
    LEARNING_RATE = 0.5
    input_dim = 1024
    output_dim = 1
    np.random.seed(0)

    def __init__(self, num_of_hidden):
        self.num_of_hidden = num_of_hidden
        self.input_layer = NeuronLayer(self.input_dim, num_of_hidden)
        self.hidden_layer = NeuronLayer(num_of_hidden, self.output_dim)
        self.output_layer = None

    def BP(self, image: np.ndarray, label: int):
        z1 = self.input_layer.calculate_net(image)
        a1 = ReLU(z1)
        z2 = self.hidden_layer.calculate_net(a1)
        res_bp = sigmoid(z2)

        delta3: np.ndarray = (res_bp[0][0] - label) * sigmoid_derivative(z2)
        delta2 = delta3.dot(self.hidden_layer.weights.T) * relu_derivative(z1)

        dw2 = a1.T.dot(delta3)
        dw1 = image.T.dot(delta2)

        d_nabla_b = [delta2, delta3]
        d_nabla_w = [dw1, dw2]

        return d_nabla_b, d_nabla_w, res_bp


class NeuronLayer:
    def __init__(self, input_dim, output_size, biases=None, weights=None):
        self.input_dim = input_dim
        self.output_size = output_size
        self.biases = biases
        self.weights = weights
        if self.biases is None and self.weights is None:
            self.biases = np.zeros((1, output_size))
            self.weights = np.random.rand((self.input_dim, output_size)) / np.sqrt(self.input_dim)

    def calculate_net(self, image_vec: np.ndarray):
        return image_vec.dot(self.weights) + self.biases


def sigmoid(x)->np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ReLU(x)->np.ndarray:
    return np.abs(x) * (x > 0)


def sigmoid_derivative(x)->np.ndarray:
    return np.exp(-x) / (1 + np.square(np.exp(-x)))


def relu_derivative(x)->np.ndarray:
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
