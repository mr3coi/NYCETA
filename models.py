import numpy as np 
# import tensorflow as tf 
import tensorflow as tf
# tf.disable_v2_behavior() 
import tensorflow.keras.backend as K
from tensorflow.compat.v1.keras.backend import set_session


def create_boro_model(num_neurons_in_layers, input_dim):
    inp = tf.keras.layers.Input(shape=(input_dim, ), name="input")
    h1 = tf.keras.layers.Dense(num_neurons_in_layers[0], activation='relu')(inp) 
    h2 = tf.keras.layers.Dense(num_neurons_in_layers[1], activation='relu')(h1) 
    o = tf.keras.layers.Dense(1, activation='relu')(h2)

    model = tf.keras.Model(inputs=inp, outputs=o) 
    # model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model



class BoroModel(object):

    def __init__(self, sess, inp_dim, num_neurons_in_layers=[100,50], learning_rate=5e-4):
        
        self.sess = sess
        set_session(self.sess)

        self.model = create_boro_model(num_neurons_in_layers, inp_dim)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

        self.gradient_loss_wrt_output = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

        self.input_gradients = tf.gradients(
            self.model.output, self.model.input, 
            grad_ys=self.gradient_loss_wrt_output)

        self.full_gradients = tf.gradients(
            self.model.output, self.model.trainable_weights,
            grad_ys=self.gradient_loss_wrt_output)

        self.optimize_op = self.optimizer.apply_gradients(
            zip(self.full_gradients, self.model.trainable_weights))

        self.sess.run(tf.compat.v1.global_variables_initializer())


    def train(self, inp, grads_loss_wrt_outputs):
        self.sess.run(self.optimize_op, feed_dict={
            self.model.input: inp,
            self.gradient_loss_wrt_output: grads_loss_wrt_outputs 
        })

    def gradients(self, inputs, grads_loss_wrt_outputs):
        gradients = self.sess.run(self.input_gradients, feed_dict={
            self.model.input: inputs, 
            self.gradient_loss_wrt_output: grads_loss_wrt_outputs 
        }) 
        return gradients[0]


    def save_model(self, path):
        tf.keras.models.save_model(self.model, path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, compile=True)

    def save_model_weights(self, path):
        self.model.save_weights(path)

    def load_model_weights(self, path):
        self.model.load_weights(path)




def create_selector_model(num_neurons_in_layers, fsize, dtsize, bridge_matrix1, bridge_matrix2):
    pu_input = tf.keras.layers.Input(shape=[fsize], name="pu_feature")
    do_input = tf.keras.layers.Input(shape=[fsize], name="do_feature")
    dt_input = tf.keras.layers.Input(shape=[dtsize], name="dt_feature")
    f = tf.keras.layers.concatenate([pu_input, do_input, dt_input])
    f = tf.keras.layers.Dense(num_neurons_in_layers[0], activation='relu')(f)
    f = tf.keras.layers.Dense(num_neurons_in_layers[1], activation='softmax')(f)

    br_matrix1 = K.constant(bridge_matrix1)
    br_matrix2 = K.constant(bridge_matrix2)
    ob1 = K.dot(f, K.transpose(br_matrix1))
    ob2 = K.dot(f, K.transpose(br_matrix2))

    o1 = tf.keras.layers.concatenate([pu_input, ob1, dt_input])
    ob = tf.keras.layers.concatenate([ob1, ob2, dt_input])
    o2 = tf.keras.layers.concatenate([ob2, do_input, dt_input])

    o = tf.keras.layers.concatenate([o1, ob, o2])

    model = tf.keras.Model(inputs=[pu_input, do_input, dt_input],  outputs=o)
    return model, pu_input, do_input, dt_input


class SelectorModel(object):

    def __init__(self, sess, num_neurons_in_layers, fsize, dtsize, 
        bridge_matrix1, bridge_matrix2, learning_rate=1e-3):
        
        self.sess = sess
        set_session(self.sess)
        self.model, self.pu_input, self.do_input, self.dt_input = create_selector_model(
                                        num_neurons_in_layers, fsize,
                                        dtsize, bridge_matrix1, bridge_matrix2)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
        self.gradient_wrt_output = tf.compat.v1.placeholder(tf.float32,
                shape=(None, 6*fsize+3*dtsize))
        
        self.full_gradients = tf.gradients(
            self.model.output, self.model.trainable_weights,
            grad_ys=self.gradient_wrt_output)

        self.optimize_op = self.optimizer.apply_gradients(
            zip(self.full_gradients, self.model.trainable_weights))

        self.sess.run(tf.compat.v1.global_variables_initializer())


    def train(self, pu_input, do_input, dt_input, grads_wrt_outputs):
        self.sess.run(self.optimize_op, feed_dict={
            self.pu_input: pu_input,
            self.do_input: do_input,
            self.dt_input: dt_input,
            self.gradient_wrt_output: grads_wrt_outputs 
        })

    def save_model(self, path):
        tf.keras.models.save_model(self.model, path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, compile=True)

    def save_model_weights(self, path):
        self.model.save_weights(path)

    def load_model_weights(self, path):
        self.model.load_weights(path)