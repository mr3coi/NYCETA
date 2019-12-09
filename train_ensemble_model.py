import argparse 
import numpy as np
import tensorflow as tf 
from sklearn.model_selection import train_test_split

from models import SelectorModel, BoroModel
from utils import create_connection
from obtain_features import extract_all_coordinates, extract_all_boroughs, get_one_hot, extract_features
from train_boro_model import FEATURE_TYPES
from bridge_info import SUPER_BRIDGES


SUPER_BOROS = {
	1: ["Manhattan", "Bronx", "EWR"], 
	2: ["Queens", "Brooklyn"],
	3: ["Staten Island"]
}

selectors = { }

boro_models = { }

br_model = None# bridge time prediction model


parser = argparse.ArgumentParser()
parser.add_argument("--db-path", type=str, default="./rides.db",
                    help="Path to the sqlite3 database file.")
parser.add_argument("--feature-type", type=int, default=2,
                    help="the type of feature vectors to be used")


def ensemble_predict(featurePU, featureDO, featureDT, selector_model, pu_model, do_model, output=None):
	prediction = selector_model.predict([featurePU, featureDO, featureDT])
	pu_input, br_input, do_input = np.hsplit(prediction, 3)

	pu_time = pu_model.model.predict(pu_input)
	br_time = br_model.model.predict(br_input)
	do_time = do_model.model.predict(do_input)
	total_time = pu_time + br_time + do_time 

	if output == None:
		return total_time
	else:
		rmse = np.sqrt(np.sum(np.square(np.subtract(total_time, output))))
		return rmse

def ensemble_train_batch(featurePU, featureDO, featureDT, output, selector_model, pu_model, do_model):
	"""A single step of the ensemble training

	:featurePU: [PULocID, PUCoords, PUBorough]
	:featureDO: [DOLocID, DOCoords, DOBorough]
	:featureDT: [PUDatetime]
	:selector_model: selector model to be used
	:pu_model: the PU point model
	:do_model: the DO point model 
	"""
	prediction = selector_model.model.predict([featurePU, featureDO, featureDT])	
	pu_input, br_input, do_input = np.hsplit(prediction, 3)

	pu_time = pu_model.model.predict(pu_input)
	br_time = br_model.model.predict(br_input)
	do_time = do_model.model.predict(do_input)
	total_time = pu_time + br_time + do_time 

	loss = np.sum(np.square(total_time-output))
	rmse = np.sqrt(loss)
	loss_grads = 2*(total_time - output )

	pu_model.train(pu_input, loss_grads)
	br_model.train(br_input, loss_grads)
	do_model.train(do_input, loss_grads)

	pu_grads = pu_model.gradients(pu_input, loss_grads)
	br_grads = br_model.gradients(br_input, loss_grads)
	do_grads = do_model.gradients(do_input, loss_grads)

	grads_wrt_sel_output = np.concatenate([pu_grads, br_grads, do_grads])

	selector_model.train(featurePU, featureDO, featureDT, grads_wrt_sel_output)

	return loss, rmse


def get_loc_vector(locId, coords, boros, includelocId=False, maxLocId=263):
	coord_vec = coords[locId].reshape((2,1))
	boros_vec = boros[locId].reshape((-1,1))
	loc_vec = np.vstack([coord_vec, boros_vec])
	if includelocId:
		locid_vec = get_one_hot([locId], 1, maxLocId)
		loc_vec = np.vstack([locid_vec, loc_vec])
	return loc_vec


def create_bridge_matrices(conn, includelocId=False):
	bridge_matrix = {}
	coords_table_name = 'coordinates'
	boros_table_name = 'locations'
	coords = extract_all_coordinates(conn, coords_table_name)
	boros = extract_all_boroughs(conn, boros_table_name)

	super_boros = [1,2,3]
	for boro in super_boros:

		bridge_matrix[boro] = {}
		for target_boro in SUPER_BRIDGES[boro]:
			bridges = SUPER_BRIDGES[boro][target_boro]
			
			for bridge_start, _ in bridges:
				vec = get_loc_vector(bridge_start, coords, boros, includelocId)
				if not target_boro in bridge_matrix[boro]:
					bridge_matrix[boro][target_boro] = vec
				else:
					bridge_matrix[boro][target_boro] = np.hstack([bridge_matrix[boro][target_boro], vec])

	return bridge_matrix



def load_cross_superboros(conn, feature_type):
	boros = [1,2,3]
	sql_batch_size = 1e5
	sql_block_size = 1e5

	PUfeatures = {}
	DOfeatures = {}
	DTfeatures = {}
	all_values = {}

	for start_boro in boros:

		PUfeatures[start_boro] = {}
		DOfeatures[start_boro] = {}
		DTfeatures[start_boro] = {}
		all_values[start_boro] = {}

		for end_boro in boros:
			# we only have to consider cross-boro trips
			if start_boro == end_boro:
				continue

			PUfeatures[start_boro][end_boro] = {'train': [], 'test': []}
			DOfeatures[start_boro][end_boro] = {'train': [], 'test': []}
			DTfeatures[start_boro][end_boro] = {'train': [], 'test': []}
			all_values[start_boro][end_boro] = {'train': [], 'test': []}
			
			data_params = {
				"conn": conn,
				"table_name": "rides",
				"variant": 'random', 
				"size": int(sql_batch_size),
				"block_size": sql_block_size,
				"datetime_onehot": FEATURE_TYPES[feature_type]['datetime_onehot'],
				"weekdays_onehot": FEATURE_TYPES[feature_type]['weekdays_onehot'],
				"include_loc_ids": FEATURE_TYPES[feature_type]['include_loc_ids'],
				"start_super_boro": SUPER_BOROS[start_boro],
				"end_super_boro": SUPER_BOROS[end_boro]
			}

			features, values = extract_features(**data_params)
			if FEATURE_TYPES[feature_type]["sparse"]:
				PUfeatures_ = features[:, :FEATURE_TYPES[feature_type]['split_indices'][0] ]
				DOfeatures_ = features[:, FEATURE_TYPES[feature_type]['split_indices'][0]:FEATURE_TYPES[feature_type]['split_indices'][1] ]
				DTfeatures_ = features[:, FEATURE_TYPES[feature_type]['split_indices'][1]: ]
			else:
				PUfeatures_, DOfeatures_, DTfeatures_ = np.hsplit(features, FEATURE_TYPES[feature_type]['split_indices'])

			total_samples = values.shape[0]
				
			# shuffle the data
			shuffle_indices = np.arange(total_samples)
			np.random.shuffle(shuffle_indices)

			PUfeatures_ = PUfeatures_[shuffle_indices]
			DOfeatures_ = DOfeatures_[shuffle_indices]
			DTfeatures_ = DTfeatures_[shuffle_indices]
			values = values[shuffle_indices]

			# get train-test split
			PUfeatures[start_boro][end_boro]['train'], PUfeatures[start_boro][end_boro]['test'], \
			DOfeatures[start_boro][end_boro]['train'], DOfeatures[start_boro][end_boro]['test'], \
			DTfeatures[start_boro][end_boro]['train'], DTfeatures[start_boro][end_boro]['test'], \
			all_values[start_boro][end_boro]['train'], all_values[start_boro][end_boro]['test'] = train_test_split(
											PUfeatures_, DOfeatures_, DTfeatures_, values, test_size=0.1, random_state=42)


	return PUfeatures, DOfeatures, DTfeatures, all_values


def batch_nn_generator(PUfeatures, DOfeatures, DTfeatures, values, batch_size):
	samples_per_epoch = values.shape[0]
	number_of_batches = samples_per_epoch/batch_size
	counter = 0
	index = np.arange(values.shape[0])
	
	while 1:
		index_batch = index[batch_size*counter: batch_size*(counter+1)]  
		PU_batch = PUfeatures[index_batch, :].todense()
		DO_batch = DOfeatures[index_batch, :].todense()
		DT_batch = DTfeatures[index_batch, :].todense()
		values_batch = values[index_batch, :]

		counter += 1
		yield np.array(PU_batch), np.array(DO_batch), np.array(DT_batch), values_batch
		if (counter > number_of_batches):
			counter=0


def save_models(epoch):
	boros = [1,2,3]
	if not os.path.isdir('ensemble'):
		os.mkdir('ensemble')
	for start_boro in boros:
		for end_boro in boros:
			path = os.path.join('ensemble', 'selectors')
			if not os.path.isdir(path):
				os.mkdir(path)
			path = os.path.join(path, f'start_{start_boro}_end_{end_boro}')
			if not os.path.isdir(path):
				os.mkdir(path)
			path = os.path.join(path, f'epoch_{epoch}')
			selectors[start_boro][end_boro].save_model(path)

	for boro in boros:
		path = os.path.join('ensemble', 'boro_models')
		if not os.path.isdir(path):
			os.mkdir(path)
		path = os.path.join(path, f'boro_{boro}')
		if not os.path.isdir(path):
			os.mkdir(path)
		path = os.path.join(path, f'epoch_{epoch}')
		boro_models[boro].save_model(path)

	path = os.path.join('ensemble', 'bridge')
	if not os.path.isdir(path):
		os.mkdir(path)
	path = os.path.join(path, f'epoch_{epoch}')
	br_model.save_model(path)


def train(PUfeatures, DOfeatures, DTfeatures, values, num_epochs=5, batch_size=1000):
	
	boros = [1,2,3]
	
	with open('ensemble_log.txt', 'a') as f:
		f.write(f'epoch, train_RMSE, val_RMSE')

	for epoch in num_epochs:

		# training
		train_RMSE = 0.0
		for start_boro in boros:
			for end_boro in boros:

				batch_gen_args = {
					"PUfeatures": PUfeatures[start_boro][end_boro]['train'],
					"DOfeatures": DOfeatures[start_boro][end_boro]['train'],
					"DTfeatures": DTfeatures[start_boro][end_boro]['train'],
					"values": values[start_boro][end_boro]['train'],
					"batch_size": batch_size
				}

				total_samples = values[start_boro][end_boro]['train'].shape[0]
				num_batches = total_samples/batch_size
				counter = 0

				for pu_batch, do_batch, dt_batch, values_batch in batch_nn_generator(**batch_gen_args):
					# train on batch
					loss, rmse = ensemble_train_batch(pu_batch, do_batch, dt_batch, values_batch, selectors[start_boro][end_boro],
						boro_models[start_boro], boro_models[end_boro])
					train_RMSE = np.sqrt(train_RMSE**2 + rmse**2)
					counter += 1
					if counter >= batch_size:
						break

		# validation
		RMSE = 0.0 
		for start_boro in boros:
			for end_boro in boros:

				batch_gen_args = {
					"PUfeatures": PUfeatures[start_boro][end_boro]['test'],
					"DOfeatures": train_DOfeatures[start_boro][end_boro]['test'],
					"DTfeatures": train_DTfeatures[start_boro][end_boro]['test'],
					"values": train_values[start_boro][end_boro]['test'],
					"batch_size": batch_size
				}

				for pu_batch, do_batch, dt_batch, values_batch in batch_nn_generator(**batch_gen_args):
					# validation
					rmse = ensemble_predict(pu_batch, do_batch, dt_batch, selectors[start_boro][end_boro],
						boro_models[start_boro], boro_models[end_boro], values_batch) 
					RMSE = np.sqrt(RMSE**2 + rmse**2)

					print(f'At epoch: {epoch}, training RMSE: {train_RMSE}, validation RMSE: {RMSE}')

					with open('ensemble_log.txt', 'a') as f:
						f.write(f'{epoch}, {train_RMSE}, {RMSE}')

		save_models(epoch)							




def main():
	global selectors
	global boro_models
	global br_model
	
	parsed_args = parser.parse_args()

	feature_type = parsed_args.feature_type
	db_name = parsed_args.db_path

	boro_model_paths = {
		1: f'final_models/Manhattan_f{feature_type}/model/',
		2: f'final_models/Queens_f{feature_type}/model/',
		3: f'final_models/Staten Island_f{feature_type}/model/'
	}

	conn = create_connection(db_name)

	# get the cross-borough features, values
	PUfeatures, DOfeatures, DTfeatures, values = load_cross_superboros(conn, feature_type)

	# get the bridge matrices for each boro
	# it is a dictionary with the following keys
	# 1: Manhattan, 2: Queens, 3: Staten Island
	bridge_matrix = create_bridge_matrices(conn)

	sess = tf.compat.v1.Session()

	# create selector networks
	fsize = PUfeatures[1][2]['train'].shape[1]
	dtsize = DTfeatures[1][2]['train'].shape[1]
	
	boros = [1,2,3]
	
	for boro in boros:
		selectors[boro] = {}
		for target_boro in boros:
			if boro == target_boro:
				continue
			selectors[boro][target_boro] = SelectorModel(sess, [100, bridge_matrix[boro][target_boro].shape[1]],
												fsize, dtsize, bridge_matrix[boro][target_boro], bridge_matrix[target_boro][boro])


	# load boro models
	feature_vec_size = FEATURE_TYPES[feature_type]['size']
	for boro in boros:
		boro_models[boro] = BoroModel(sess, feature_vec_size, [200, 50])
		boro_models[boro].load_model(boro_model_paths[boro])

	# create bridge model
	br_model = BoroModel(sess, feature_vec_size, [100, 50])

	# train 


if __name__ == "__main__":
	main()