import numpy as np 
import tensorflow as tf 

class BoroModel(tf.keras.Model):

	def __init__(self, num_neurons_in_layers=[100,50]):
		super(BoroModel, self).__init__()
		self.dense1 = tf.keras.layers.Dense(num_neurons_in_layers[0], 
				activation=tf.nn.relu)
		self.dense2 = tf.keras.layers.Dense(num_neurons_in_layers[1],
				activation=tf.nn.relu)
		self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.relu)


	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		y = self.dense3(x)
		return y


class SelectorModel(tf.keras.Model):

	def __init__(self, num_neurons_in_layers=[50,9]):
		super(SelectorModel, self).__init__()
		self.dense1 = tf.keras.layers.Dense(num_neurons_in_layers[0],
				activation=tf.nn.relu)
		self.dense2 = tf.keras.layers.Dense(num_neurons_in_layers[1],
				activation=tf.nn.softmax)
		self.output = tf.keras.backend.argmax(self.dense2)


	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		y = self.output(x)
		return y
		