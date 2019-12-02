import os
import datetime
import argparse
import numpy as np
from models import BoroModel
from utils import create_connection
from obtain_features import extract_features

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--db-path", type=str, default="./rides.db",
                    help="Path to the sqlite3 database file.")



def train_on_batches(model, data_generator, data_gen_args):

	model_dir = os.path.join("models",  datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.mkdir(model_dir)

	save_every = 5
	num_epochs = 200
	optimizer = 'Adam'
	loss = tf.keras.losses.MSE
	model.compile(optimizer=optimizer, loss=loss)

	iter_num = 0
	log_file = open(os.path.join(model_dir, "training_logs.txt"), "w")

	for epoch in range(num_epochs):
		
		print(f"\nEpoch {epoch+1} begins")
		epochLoss = 0.0
		for features, outputs in data_generator(**data_gen_args):
			iter_num += 1
			print(f'Training epoch: {epoch+1}, iteration: {iter_num}')
			if not isinstance(features,np.ndarray):
				features = features.toarray()
			if not isinstance(outputs, np.ndarray):
				outputs = outputs.toarray()
			MSEloss = model.train_on_batch(x=features, y=outputs)
			print(f'loss: {MSEloss}')
			epochLoss += MSEloss
			log_file.write(f'{epoch+1}, {iter_num}, {MSEloss}\n')

			if iter_num % save_every == 0:
				tf.keras.models.save_model(model, os.path.join(model_dir, f'model_{iter_num}'))

		print(f"Total Epoch Loss = {epochLoss}")
		print("======================================================================\n")

	log_file.close()


def train(model, data_generator, data_gen_args):
	
	# features, values = data_generator(**data_gen_args)
	features = np.load("features_queens.npy", allow_pickle=True)
	values = np.load("outputs_queens.npy", allow_pickle=True)
	model_dir = os.path.join("models",  datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.mkdir(model_dir)

	optimizer = 'Adam'
	loss = tf.keras.losses.MSE
	model.compile(optimizer=optimizer, loss=loss)

	num_epochs = 200
	model.fit(features, values, epochs=num_epochs, verbose=2)

	tf.keras.models.save_model(model, os.path.join(model_dir, f'model_{num_epochs}'))


def main():
	parsed_args = parser.parse_args()

	model = BoroModel([200,50])
	# conn = create_connection(parsed_args.db_path)

	conn = None
	batch_size = 2048
	block_size = 512
	variant = 'all'

	data_generator = extract_features
	data_generator_arguments = {
			"conn": conn,
			"table_name": "rides",
			"variant": variant, 
			"size": batch_size,
			"block_size": block_size,
			"datetime_onehot": False,
			"weekdays_onehot": True,
			"include_loc_ids": True
		}

	if variant == 'all':
		train(model, data_generator, data_generator_arguments)
	elif variant == 'batch':
		train_on_batches(model, data_generator, data_generator_arguments)




if __name__ == "__main__":
	main()