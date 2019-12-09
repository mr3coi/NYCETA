import os
import datetime
import argparse
import numpy as np
from scipy import sparse
from models import BoroModel, create_boro_model
from utils import create_connection
from obtain_features import extract_features

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger


SUPER_BOROS = [
	["Manhattan", "Bronx", "EWR"], 
	["Queens", "Brooklyn"],
	["Staten Island"]
]

FEATURE_TYPES = {
	1:	{
			"datetime_onehot": False,
			"weekdays_onehot": False,
			"include_loc_ids": False,
			"size": 22
		},
	2:	{
			"datetime_onehot": True,
			"weekdays_onehot": True,
			"include_loc_ids": False,
			"size": 210
		},
	3:	{
			"datetime_onehot": True,
			"weekdays_onehot": True,
			"include_loc_ids": True,
			"size": 736
		}
}


parser = argparse.ArgumentParser()
parser.add_argument("--db-path", type=str, default="./rides.db",
                    help="Path to the sqlite3 database file.")
parser.add_argument("--variant", type=str, default="all",
                    help="'all' or 'batch'")
parser.add_argument("--saved", default=False, action='store_true',
                    help="whether we have pre saved features or not")
parser.add_argument("--feature-type", type=int, default=1,
                    help="the type of feature vectors to be used")
parser.add_argument("--superboro-id", type=int, default=2,
                    help="the id of the super boro to be trained")
parser.add_argument("--sparse", default=False, action='store_true',
                    help="whether we have sparse features or not")
parser.add_argument("--num-epochs", type=int, default=20,
                    help="the number of epochs to be trained")
parser.add_argument("--batch-size", type=int, default=1000,
                    help="the batch size to be used for trained")
parser.add_argument("--model-path", type=str, default=None,
                    help="Path to the model to hot start from")


TOTAL_ENTRIES_IN_DB = 67302302


def nn_batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(X_batch),y_batch
        if (counter > number_of_batches):
            counter=0


def train_on_batches(model, data_generator, data_gen_args, saved, features_file, values_file,
				model_dir, isSparse=True, num_epochs=20):

	if not saved:
		features, values = data_generator(**data_gen_args)
		if isSparse:
			sparse.save_npz(features_file, features)
		else:
			np.save(features_file, features)
		np.save(values_file, values)
	else:
		if isSparse:
			print(f"Loading sparse features from {features_file}")
			features = sparse.load_npz(features_file+".npz")
		else:
			print(f"Loading dense features from {features_file}")
			features = np.load(features_file, allow_pickle=True)
		
		print(f"Loading output values from {values_file}")
		values = np.load(values_file, allow_pickle=True)

	total_samples = features.shape[0]

	train_features, test_features, train_values, test_values = train_test_split(
			features, values, test_size=0.1, random_state=42)

	save_every = 5
	batch_size = 1000
	steps_per_epoch = int(total_samples/batch_size)+1
	

	optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
	model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
		 metrics=[tf.keras.metrics.RootMeanSquaredError()])

	csv_logger = CSVLogger(os.path.join(model_dir, 'log.csv'), append=True, separator=';')

	model.fit_generator(nn_batch_generator(train_features, train_values, batch_size=int(batch_size)), 
						epochs=num_epochs, verbose=2, steps_per_epoch=steps_per_epoch,
						use_multiprocessing=True, workers=4, callbacks=[csv_logger])

	test_scores = model.evaluate(test_features, test_values, verbose=0)
	print(f"Model evalutaion on test data\n {test_scores}")

	tf.keras.models.save_model(model, os.path.join(model_dir, f'model_{num_epochs}'))

	# iter_num = 0
	# log_file = open(os.path.join(model_dir, "training_logs.txt"), "w")

	# for epoch in range(num_epochs):
		
	# 	print(f"\nEpoch {epoch+1} begins")
	# 	epochLoss = 0.0
	# 	for features, outputs in data_generator(**data_gen_args):
	# 		iter_num += 1
	# 		print(f'Training epoch: {epoch+1}, iteration: {iter_num}')
	# 		if not isinstance(features,np.ndarray):
	# 			features = features.toarray()
	# 		if not isinstance(outputs, np.ndarray):
	# 			outputs = outputs.toarray()
	# 		MSEloss = model.train_on_batch(x=features, y=outputs)
	# 		print(f'loss: {MSEloss}')
	# 		epochLoss += MSEloss
	# 		log_file.write(f'{epoch+1}, {iter_num}, {MSEloss}\n')

	# 		if iter_num % save_every == 0:
	# 			tf.keras.models.save_model(model, os.path.join(model_dir, f'model_{iter_num}'))

	# 	print(f"Total Epoch Loss = {epochLoss}")
	# 	print("======================================================================\n")

	# log_file.close()


def train(model, data_generator, data_gen_args, saved, features_file, values_file, 
			model_dir, isSparse=False, num_epochs=20, batch_size=1000, start_epoch=1):
	
	if not saved:
		features, values = data_generator(**data_gen_args)
		if isSparse:
			sparse.save_npz(features_file, features)
		else:
			np.save(features_file, features)
		np.save(values_file, values)
	else:
		if isSparse:
			print(f"Reading sparse features from {features_file}.npz")
			features = sparse.load_npz(features_file+".npz")
			features = features.toarray()
		else:
			print(f"Reading dense features from {features_file}")
			features = np.load(features_file, allow_pickle=True)
		print(f"Reading values from {values_file}")
		values = np.load(values_file, allow_pickle=True)

	train_features, test_features, train_values, test_values = train_test_split(
			features, values, test_size=0.1, random_state=42)

	optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
	model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
		 metrics=[tf.keras.metrics.RootMeanSquaredError()])

	csv_logger = CSVLogger(os.path.join(model_dir, 'log.csv'), append=True, separator=';')

	model.fit(train_features, train_values, epochs=num_epochs, batch_size=batch_size,
				validation_split=0.1, verbose=1, callbacks=[csv_logger], initial_epoch=start_epoch)

	test_scores = model.evaluate(test_features, test_values, verbose=0, callbacks=[csv_logger])
	print(f"Model evalutaion on test data\n {test_scores}")

	with open(os.path.join(model_dir, 'eval.txt'), 'w') as f:
		f.write(str(test_scores))

	tf.keras.models.save_model(model, os.path.join(model_dir, f'model_{num_epochs}'))


def main():
	parsed_args = parser.parse_args()

	conn = create_connection(parsed_args.db_path, check_same_thread=False)

	feature_type = parsed_args.feature_type
	saved = parsed_args.saved
	variant = parsed_args.variant
	superboro_id = parsed_args.superboro_id
	is_sparse = parsed_args.sparse
	num_epochs = parsed_args.num_epochs
	batch_size = parsed_args.batch_size
	model_path = parsed_args.model_path

	feature_vec_size = FEATURE_TYPES[feature_type]['size']
	super_boro = SUPER_BOROS[superboro_id]

	if not model_path is None:
		model = tf.keras.models.load_model(model_path)
		start_epoch = int(model_path.split("_")[-1])
	else:
		model = create_boro_model([200,50], feature_vec_size)
		start_epoch = 0

	dt_now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	model_dir = os.path.join("models",  f'{dt_now}_{super_boro[0]}_f{feature_type}')
	os.mkdir(model_dir)

	sql_batch_size = 1e6
	sql_block_size = 1e5

	data_generator = extract_features
	data_generator_arguments = {
			"conn": conn,
			"table_name": "rides",
			"variant": 'all', 
			"size": sql_batch_size,
			"block_size": sql_block_size,
			"datetime_onehot": FEATURE_TYPES[feature_type]['datetime_onehot'],
			"weekdays_onehot": FEATURE_TYPES[feature_type]['weekdays_onehot'],
			"include_loc_ids": FEATURE_TYPES[feature_type]['include_loc_ids'],
			"start_super_boro": super_boro,
			"end_super_boro": super_boro
		}

	features_file = os.path.join('data', f'features_{super_boro[0]}_{feature_type}.npy')
	values_file = os.path.join('data', f'values_{super_boro[0]}_{feature_type}.npy')

	if variant == 'all':
		train(model, data_generator, data_generator_arguments, saved, features_file, 
				values_file, model_dir, is_sparse, num_epochs, batch_size, start_epoch)
	elif variant == 'batch':
		train_on_batches(model, data_generator, data_generator_arguments, saved,
				features_file, values_file, model_dir, is_sparse, num_epochs, start_epoch)




if __name__ == "__main__":
	main()


# feature vector sizes
# 
# 1. no one hot, no loc id = 22
# 2. both one hot, no loc id = 210
# 3. both one hot, with loc id