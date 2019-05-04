import numpy as np

from loadDataSet import load_datad_set_from_folder


class DNN:
    LEARNING_RATE = 0.05
    input_dim = 1024
    output_dim = 1
    np.random.seed(0)
    df_val = load_datad_set_from_folder(r"C:\Users\Yotam\Desktop\MS_Dataset_2019\validation\\")

    def __init__(self, num_of_hidden):
        self.num_of_hidden = num_of_hidden
        self.input_layer = NeuronLayer(self.input_dim, num_of_hidden)
        self.hidden_layer = NeuronLayer(num_of_hidden, self.output_dim)
        self.output_layer = None

    def train(self, folder):
        df = load_datad_set_from_folder(folder)
        acc = 0.5
        while acc < 0.95:
            loss, acc = self.update_weight_bias(df, self.LEARNING_RATE, len(df))
            print('Validation:` ',self.run_test())
            print(acc)
        STOP = True

    def run_test(self):
        df = self.df_val
        accuracy = []
        for index, row in df.iterrows():
            res = self.predict(row['data'])
            accuracy.append(bool(row['lable']) == res)
        return np.mean(accuracy)

    def predict(self, image):
        z1 = self.input_layer.calculate_net(image)
        a1 = ReLU(z1)
        z2 = self.hidden_layer.calculate_net(a1)
        res_bp = sigmoid(z2)
        return res_bp > 0.5

    def BP(self, image: np.ndarray, label: int):
        z1 = self.input_layer.calculate_net(image)
        a1 = ReLU(z1)
        z2 = self.hidden_layer.calculate_net(a1)
        res_bp = sigmoid(z2)
        # print(res_bp)
        delta3: np.ndarray = (res_bp[0][0] - label) * sigmoid_derivative(z2)
        delta2 = delta3.dot(self.hidden_layer.weights.T) * relu_derivative(z1)
        dw2 = a1.T.dot(delta3)
        dw1 = image.T.dot(delta2)

        d_nabla_b = [delta2, delta3]
        d_nabla_w = [dw1, dw2]

        return d_nabla_b, d_nabla_w, res_bp

    def update_weight_bias(self, data_set, lr, batch_size):
        nabla_w = [np.zeros(w.shape) for w in [self.input_layer.weights, self.hidden_layer.weights]]
        nabla_b = [np.zeros(b.shape) for b in [self.input_layer.biases, self.hidden_layer.biases]]
        loss = []
        accuracy = []
        for index, row in data_set.iterrows():
            db, dw, prob = self.BP(row['data'], row['lable'])
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, dw)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, db)]
            loss.append((row['lable'] - prob[0][0]) ** 2 / 2)
            accuracy.append(bool(row['lable']) == (prob[0][0] > 0.5))

        self.input_layer.weights, self.hidden_layer.weights = [w - (lr / batch_size) * dw for w, dw in zip(
            [self.input_layer.weights, self.hidden_layer.weights], nabla_w)]
        self.input_layer.biases, self.hidden_layer.biases = [b - (lr / batch_size) * db for b, db in
                                                             zip([self.input_layer.biases, self.hidden_layer.biases],
                                                                 nabla_b)]
        mean_loss = np.mean(loss)
        mean_accuracy = np.mean(accuracy)
        return mean_loss, mean_accuracy


class NeuronLayer:
    def __init__(self, input_dim, output_size, biases=None, weights=None):
        self.input_dim = input_dim
        self.output_size = output_size
        self.biases = biases
        self.weights = weights
        if self.biases is None and self.weights is None:
            self.biases = np.zeros((1, output_size))
            self.weights = np.random.randn(self.input_dim, output_size) / np.sqrt(self.input_dim)

    def calculate_net(self, image_vec: np.ndarray):
        return image_vec.dot(self.weights) + self.biases


def sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ReLU(x) -> np.ndarray:
    return np.abs(x) * (x > 0)


def sigmoid_derivative(x) -> np.ndarray:
    return np.exp(-x) / (1 + np.square(np.exp(-x)))


def relu_derivative(x) -> np.ndarray:
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
