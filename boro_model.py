import numpy as np 
import tensorflow as tf 
from tf.keras.layers import Dense

class BoroModel(tf.keras.Model):

	def __init__(self, num_neurons_in_layers=[100,50]):
		super(BoroModel, self).__init__()
		self.dense1 = Dense(num_neurons_in_layers[0], 
				activation=tf.nn.relu)
		self.dense2 = Dense(num_neurons_in_layers[1],
				activation=tf.nn.relu)
		self.dense3 = Dense(1, activation=tf.nn.relu)


	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		y = self.dense3(y)
		return y
