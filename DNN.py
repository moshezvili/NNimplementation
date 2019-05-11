import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os
import matplotlib.image as mpimg

def load_data_set_from_folder(folder):
	filesList = os.listdir(folder)
	df = pd.DataFrame()
	for file in filesList:
		im = mpimg.imread(folder+ file)
		im2 = np.ravel(im).reshape((1,1024))
		word_lable = file.split("_")[0]
		if word_lable == 'neg':
			lable = 0
		else:
			lable = 1
		dfRow  = pd.DataFrame({'data': [im2] , 'lable' : [lable]})
		df  = df.append(dfRow)
	return df


class DrawRealTime:
	def __init__(self, epochs):
		plt.ion()
		self.fig, self.ax = plt.subplots()
		self.x, self.y, self.z = [], [], []
		plt.xlim(0, epochs)
		plt.ylim(0, 1)
		plt.draw()

	def updatePlot(self, epoch, acc_train, acc_val):
		self.x.append(epoch)
		self.y.append(acc_train)
		self.z.append(acc_val)
		self.ax.plot(self.x, self.y,color='blue')
		self.ax.plot(self.x, self.z,color='green')
		plt.legend(["train","validation"])
		self.fig.canvas.draw_idle()
		plt.pause(0.1)


class DNN:
	LEARNING_RATE = 0.01
	input_dim = 1024
	output_dim = 1
	BATCH_SIZE = 64
	np.random.seed(0)
	df_val = load_data_set_from_folder(r"C:\Users\Yotam\Desktop\MS_Dataset_2019\validation\\")

	def __init__(self, num_of_hidden):
		self.num_of_hidden = num_of_hidden
		self.input_layer = NeuronLayer(self.input_dim, num_of_hidden)
		self.hidden_layer = NeuronLayer(num_of_hidden, self.output_dim)
		self.output_layer = None

	def train(self, folder: str, learning_rate : float =0.01 ,batch_size : int = 64, epochs: int = 250, is_plot : bool = False):
		trainDf = pd.DataFrame(columns=['epoch','train_acc','val_acc'])
		if is_plot:
			accPlot = DrawRealTime(epochs)
		df = load_data_set_from_folder(folder)
		if epochs is not None:
			for i in range(epochs):
				acc, loss = 0, 1
				tic = time.time()
				batches_list = self.divide_data_set_to_batches(df, batch_size)
				for batch in batches_list:
					loss, acc = self.update_weight_bias(batch,learning_rate, batch_size)
				toc = time.time()
				acc_val = self.run_test()
				if is_plot:
					accPlot.updatePlot(i, acc, acc_val)
				print(f'Epoch {i + 1}/{epochs}')
				print(f"{toc - tic}s - loss: {loss} - acc: {acc}")
				print('Validation:', acc_val)
				dfRow = pd.DataFrame({'epoch' : [i], 'train_acc' : [acc], 'val_acc' : [acc_val]})
				trainDf = trainDf.append(dfRow)
			trainDf.to_csv("C:\\Users\\Yotam\\Desktop\\gridsearch\\" + str(self.num_of_hidden)+'_'+str(batch_size)+'_'+str(learning_rate)+'.csv',index=False)
		else:
			acc = 0.5
			while acc < 0.95:
				tic = time.time()
				batches_list = self.divide_data_set_to_batches(df, self.BATCH_SIZE)
				for batch in batches_list:
					loss, acc = self.update_weight_bias(batch, self.LEARNING_RATE, self.BATCH_SIZE)
				toc = time.time()
				print(f"{toc - tic}s - loss: {loss} - acc: {acc}")
				print('Validation:` ', self.run_test())

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

	def divide_data_set_to_batches(self, data_set, batch_size):
		bathcList = []
		data_set = data_set.reset_index().drop(columns=['index'])
		while len(data_set) > 0:
			batch = data_set.sample(batch_size, random_state=1)
			data_set = data_set.drop(batch.index)
			bathcList.append(batch)
		return bathcList

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


def create_model_and_train(num_hiddens : int =  512, learning_rate : float =0.01 ,batch_size : int = 64, epochs: int = 250):
	dnn = DNN(num_hiddens)
	dnn.train('C:\\Users\\Yotam\\Desktop\\MS_Dataset_2019\\training\\', learning_rate=learning_rate,batch_size=batch_size,epochs=epochs)

def grid_search(num_hidden_list , batch_size_list ,learning_list):
	nOfProcessors=4
	with ProcessPoolExecutor(max_workers=nOfProcessors) as executor:
		for num_hidden in num_hidden_list:
			for lr in learning_list:
				for batch_size in batch_size_list:
					executor.submit(create_model_and_train,num_hidden,lr,batch_size,150)
if __name__ == '__main__':
	num_hidden_list = [1000]
	batch_size_list = [16,32]
	learning_list = [0.01,0.02]
	grid_search(num_hidden_list , batch_size_list ,learning_list)