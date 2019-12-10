import os
import tensorflow as tf 
from models import create_boro_model

modelpath = '/home/amritsin/Desktop/10-701/project/NYCETA/models/2019-12-09_21-51-29_Manhattan_f2'

weights_path = os.path.join(modelpath, 'weights')
weights_path = os.path.join(weights_path, 'weights_00000005.h5')

model = create_boro_model([200, 50], 210)

model.load_weights(weights_path)

tf.keras.models.save_model(model, os.path.join(modelpath, 'model_5'))
