import numpy as np 
import tensorflow as tf 
import tf.keras.backend as K


def create_boro_model(num_neurons_in_layers, input_dim):
	inp = tf.keras.layers.Input(shape=(input_dim, ), name="input")
	h1 = tf.keras.layers.Dense(num_neurons_in_layers[0], activation='relu')(inp) 
	h2 = tf.keras.layers.Dense(num_neurons_in_layers[1], activation='relu')(h1) 
	o = tf.keras.layers.Dense(1, activation='relu')(h2)

	model = tf.keras.Model(inputs=inp, outputs=o) 
	# model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
	return model


class BoroModel(object):

	def __init__(self, sess, inp_dim, num_neurons_in_layers=[100,50], learning_rate=1e-3):
		
		self.sess = sess
		K.set_session(self.sess)
		self.model = create_boro_model(num_neurons_in_layers, inp_dim)

		self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

		self.gradient_loss_wrt_output = tf.placeholder(tf.float32, shape=())

		self.input_gradients = tf.gradients(
			self.model,output, self.model.input, 
			grad_ys=self.gradient_loss_wrt_output)

		self.full_gradients = tf.gradients(
        	self.model.output, self.model.trainable_weights,
       		grad_ys=self.gradient_loss_wrt_output)

		self.optimizer_op = self.optimizer.apply_gradients(
			zip(self.full_gradients, self.model.trainable_weights))

        self.sess.run(tf.initialize_all_variables())


    def train(self, inp, grads_loss_wrt_outputs):
		self.sess.run(self.optimize_op, feed_dict={
	        self.model.input: inp,
	        self.gradient_wrt_output: grads_loss_wrt_outputs 
	    })

    def gradients(self, inputs, grads_loss_wrt_outputs):
    	gradients = self.sess.run(self.input_gradients, feed_dict={
    		self.model.input: inputs, 
    		self.gradient_loss_wrt_outputs}) 
    	return gradients[0]


    def save_model(self, path):
    	tf.keras.models.save_model(self.model, path)


    def load_model(self, path):
    	self.model = tf.keras.models.load_model(path, compile=True)




def create_selector_model(num_neurons_in_layers, fsize, dtsize, bridge_matrix1, bridge_matrix2):
	pu_input = tf.keras.layers.Input(shape=[fsize], name="pu_feature")
	do_input = tf.keras.layers.Input(shape=[fsize], name="do_feature")
	dt_input = tf.keras.layers.Input(shape=[dtsize], name="dt_feature")
	f = tf.keras.layers.concatenate()([pu_input, do_input, dt_input])
	f = tf.keras.layers.Dense(num_neurons_in_layers[0], activation='relu')(f)
	f = tf.keras.layers.Dense(num_neurons_in_layers[1], activation='softmax')(f)

	br_matrix1 = K.variable(bridge_matrix, trainable=False)
	br_matrix2 = K.variable(bridge_matrix, trainable=False)
	ob1 = K.dot(K.transpose(f), br_matrix1)
	ob2 = K.dot(K.transpose(f), br_matrix2)

	o1 = tf.layers.concatenate([pu_input, ob1, dt_input])
	ob = tf.layers.concatenate([ob1, ob2, dt_input])
	o2 = tf.layers.concatenate([ob2, do_input, dt_input])

	o = tf.layers.concatenate([o1, ob, o2])

	model = tf.keras.Model(inputs=[pu_input, do_input, dt_input],  outputs=o)
	return model


class SelectorModel(object):

	def __init__(self, sess, num_neurons_in_layers, fsize, dtsize, 
		bridge_matrix1, bridge_matrix2, learning_rate):
		
		K.set_session(sess)
		self.model = create_selector_model(num_neurons_in_layers, fsize,
						dtsize, bridge_matrix1, bridge_matrix2)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
		
		self.gradient_wrt_output = tf.placeholder(tf.float32,
				shape=(None, 6*fsize+3*dsize))
		
		self.full_gradients = tf.gradients(
        	self.model.output, self.model.trainable_weights,
       		grad_ys=self.gradient_wrt_output)

		self.optimizer_op = self.optimizer.apply_gradients(
			zip(self.full_gradients, self.model.trainable_weights))


	def train(self, pu_input, do_input, dt_input, grads_wrt_outputs):
		self.sess.run(self.optimize_op, feed_dict={
	        self.model.input: [pu_input, do_input, dt_input],
	        self.gradient_wrt_output: grads_wrt_outputs 
	    })


		