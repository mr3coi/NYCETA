import numpy as np 
import tensorflow as tf 
from models import SelectorModel


selectors = [
	SelectorModel(),
	SelectorModel(),
	SelectorModel()
]

boro_models = [
	
]

br_model = 


def ensemble_train_batch(featurePU, featureDO, featureDT, output, sel_model, PU_model, DO_model):
	"""A single step of the ensemble training

	:featurePU: [PULocID, PUCoords, PUBorough]
	:featureDO: [DOLocID, DOCoords, DOBorough]
	:featureDT: [PUDatetime]
	:sel_model: (int) which selector model is to be used
	:PU_model: which is the PU point model
	:DO_model: which is the DO point model 
	"""
	selector_model = selectors[sel_model]

	prediction = selector_model.model.predict([featurePU, featureDO, featureDT])	
	pu_input, br_input, do_input = np.hsplit(prediction, 3)

	pu_model = boro_models[PU_model]
	do_model = boro_models[DO_model]

	pu_time = pu_model.model.predict(pu_input)
	br_time = br_model.model.predict(br_input)
	do_time = do_model.model.predict(do_input)
	total_time = pu_time + br_time + do_time 

	loss_grads = 2*(total_time - output )

	pu_model.train(pu_input, loss_grads)
	br_model.train(br_input, loss_grads)
	do_model.train(do_input, loss_grads)

	pu_grads = pu_model.gradients(pu_input, loss_grads)
	br_grads = br_model.gradients(br_input, loss_grads)
	do_grads = do_model.gradients(do_input, loss_grads)

	grads_wrt_sel_output = np.concatenate([pu_grads, br_grads, do_grads])

	selector_model.train(featurePU, featureDO, featureDT, grads_wrt_sel_output)

