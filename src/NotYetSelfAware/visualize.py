import numpy as np
from math import cos, sin, atan
from matplotlib import pyplot as plt

# https://stackoverflow.com/a/33720100/10964320

class Neuron():
	def __init__(self, x, y, neuron_radius, ax):
		self.x = x
		self.y = y
		self.neuron_radius = neuron_radius
		self.ax = ax

	def draw(self):
		circle = plt.Circle(
			(self.x, self.y), radius=self.neuron_radius, fill=True, edgecolor="black",facecolor="white")
		self.ax.add_patch(circle)
		# plt.gca().add_patch(circle)


class Layer():
	def __init__(self, network, number_of_neurons, weights, const, ax):
		self.ax = ax
		self.const = const
		self.previous_layer = self.__get_previous_layer(network)
		self.y = self.__calculate_layer_y_position()
		self.neurons = self.__intialise_neurons(number_of_neurons)
		self.weights = weights
		self.lines = []

	def __intialise_neurons(self, number_of_neurons):
		neurons = []
		x = self.__calculate_left_margin_so_layer_is_centered(
			number_of_neurons)
		for iteration in range(number_of_neurons):
			neuron = Neuron(x, self.y, self.const['neuron_radius'], self.ax)
			neurons.append(neuron)
			x += self.const['horizontal_distance_between_neurons']
		return neurons

	def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
		return self.const['horizontal_distance_between_neurons'] * \
			(self.const['number_of_neurons_in_widest_layer'] - number_of_neurons) / 2

	def __calculate_layer_y_position(self):
		if self.previous_layer:
			return self.previous_layer.y + self.const['vertical_distance_between_layers']
		else:
			return 0

	def __get_previous_layer(self, network):
		if len(network.layers) > 0:
			return network.layers[-1]
		else:
			return None

	def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
		angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
		x_adjustment = self.const['neuron_radius'] * sin(angle)
		y_adjustment = self.const['neuron_radius'] * cos(angle)
		line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
		line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
		color = "r" if linewidth <= 0 else "b"
		linewidth = linewidth * linewidth
		max_linewidth = 10
		if linewidth >= max_linewidth:
			linewidth = max_linewidth
		elif linewidth <= -max_linewidth:
			linewidth = -max_linewidth
		line = plt.Line2D(line_x_data, line_y_data, linewidth=linewidth, c=color)
		self.ax.add_line(line)
		# self.lines.append(line)
		# plt.gca().add_line(line)

	def clean(self):
		for l in self.lines:
			l.remove()
		self.lines = []

	def draw(self):
		for this_layer_neuron_index in range(len(self.neurons)):
			neuron = self.neurons[this_layer_neuron_index]
			if self.previous_layer:
				for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
					previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
					weight = self.previous_layer.weights[this_layer_neuron_index,
														 previous_layer_neuron_index]
					self.__line_between_two_neurons(
						neuron, previous_layer_neuron, weight)
			neuron.draw()


class NeuralNetworkVisu():
	def __init__(self):
		self.layers = []
		self.const = {
			'vertical_distance_between_layers' : 6,
			'horizontal_distance_between_neurons' : 2,
			'neuron_radius' : 0.2,
			'number_of_neurons_in_widest_layer' : -1,
		}
		self.hori_size = 20
		self.vert_size = 8 + 4
		self.fig = plt.figure(
			figsize=(self.hori_size, self.vert_size), facecolor='whitesmoke')
		grid = plt.GridSpec(2, 4, wspace=0.1, hspace=0.2)
		self.ax_network = self.fig.add_subplot(grid[:, :2])
		self.ax_losses = self.fig.add_subplot(grid[0, 2:])
		# self.ax_accues = self.ax_losses.twinx()
		self.loss_color = "tab:red"
		self.acc_color = "tab:blue"
		self.ax_accues = self.fig.add_subplot(grid[1, 2:])
		plt.ion()

	def add_layer(self, number_of_neurons, weights=None):
		if self.const['number_of_neurons_in_widest_layer'] < number_of_neurons:
			self.const['number_of_neurons_in_widest_layer'] = number_of_neurons
			self.const['horizontal_distance_between_neurons'] = self.hori_size / (number_of_neurons + 1)

		layer = Layer(self, number_of_neurons, weights, self.const, self.ax_network)
		self.layers.append(layer)
		self.const['vertical_distance_between_layers'] = self.vert_size / (len(self.layers))

	def update_weights(self, layers, losses, accues):
		for v, l in zip(self.layers, layers):
			v.weights = l.params['W']
		self.losses = losses
		self.accues = accues
			# v.weights = l.grads['dW']

	def draw(self, e):
		self.ax_network.clear()
		self.ax_losses.clear()
		self.ax_accues.clear()

		for layer in self.layers:
			# layer.clean()
			layer.draw()
		self.ax_network.axis('scaled')
		self.ax_network.axis('off')
		self.ax_network.set_title("Neural Network")
		# self.ax_network.set_facecolor('red')

		# self.ax_losses.set_xlabel('Epochs')

		self.ax_losses.set_ylabel("Loss", color=self.loss_color)
		self.ax_losses.plot(self.losses, label="Loss", color=self.loss_color)
		self.ax_losses.tick_params(axis='y', labelcolor=self.loss_color)
		# self.ax_losses.set_facecolor('lightcyan')
		# self.ax_losses.axis('scaled')
		
		# self.ax_losses.set_title("Losses")

		# self.ax_accues.yaxis.tick_right()
		self.ax_accues.set_ylabel("Accuracy", color=self.acc_color)
		self.ax_accues.plot(self.accues, label="Accuracy", color=self.acc_color)
		self.ax_accues.tick_params(axis='y', labelcolor=self.acc_color)
		self.ax_accues.set_xlabel('Epochs')

		# self.ax_accues.set_title("Accuracy")
		plt.pause(1e-4)
		# self.ax_accues.axis('scaled')
		# self.fig.savefig(f"my_img.png", format="png")
		# self.fig.savefig(f"./assets/raw_img/{e:07}.png", format="png")
		plt.show()
		plt.pause(1e-4)
		# plt.ioff()
		# plt.ion()

	def exit(self):
		plt.close()
		plt.ioff()


if __name__ == "__main__":
	network = NeuralNetworkVisu()
	weights1 = np.array([
		[0, 0, 0.5, 0, 0, 0, 0, 0, 1, 1],
		[0, 0, -0.2, 0, 1, 2, 1, 3, 0, 0],
		[0, 0, 1, -2, 0, 0, 1, -1, 0, 0],
		[0, 1, 0, 1, -1, 1, 0, 1, 0, 1]])
	network.add_layer(10, weights1)
	network.add_layer(4)
	# network.add_layer(5)
	# network.add_layer(1)
	network.draw()
