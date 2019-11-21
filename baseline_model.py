import numpy as np
from scipy import sparse
import argparse
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import xgboost as xgb

import sys
from time import time
import os
from datetime import datetime as dt
from pytz import timezone

from obtain_features import *

parser = argparse.ArgumentParser(description="Specify baseline model to train")
parser.add_argument('-m', "--model", type=str, default="xgboost",
					choices = ["gbrt", "xgboost", "lightgbm", "stc_xgboost"],
					help="Choose which baseline model to train (default: xgboost)")
parser.add_argument('--db-path', type=str, default="./rides.db",
					help="Path to the sqlite3 database file.")
parser.add_argument('-r', '--rand-subset', type=int, default=0,
					help="Shuffle the dataset, then sample a subset "
						 "with size specified by argument (default: 0). "
						 "Size 0 means the whole dataset is used (i.e. variant=all)")
parser.add_argument('--batch-size', type=int, default=-1,
					help="Batch size for semi-random batch learning")
parser.add_argument('--num-trees', type=int, default=100,
					help="Number of trees (iterations) to train")
parser.add_argument('--max-depth', type=int, default=3,
					help="The maximum depth of each regression tree")
parser.add_argument('-v', '--verbose', action='store_true',
					help="Let the model print training progress (if supported)")
parser.add_argument('-lr', '--learning-rate', type=float, default=0.1,
					help="The learning rate to train the model with")
parser.add_argument('-ssr', '--subsample-rate', type=float, default=1,
					help="Subsampling rate for rows. "
						 "Must be between 0 and 1.")
parser.add_argument('--log', action='store_true',
					help="Record training log into a text file "
						 "(default location: 'NYCETA/logs')")


def write_log(args, stats, dirname="logs"):
	'''Store a log of losses per each iteration onto a file.
	Creates a directory under current pwd named {dirname},
	and stores hyperparameters as well as loss/objective values
	depending on which are available in 'stats' dictionary.

	:args: Argparse parsed object 
	:stats: dictionary containing training statistics
	:dirname: name of the directory to contain log files
	:returns: None
	'''
	def log_dir(dirname):
		'''Creates a directory in project root directory
		to store training logs, unless already exists.

		:dirname: the name of the directory
		:returns: the path to the directory
		'''
		log_path = os.path.join(os.getcwd(),dirname)
		try:
			os.mkdir(log_path)
		except OSError:
			pass
		return log_path

	curr_time = dt.now().astimezone(timezone('US/Eastern')).strftime("%Y-%m-%d-%H-%M-%S")
	log_addr = os.path.join(log_dir(dirname), f'log_{curr_time}.txt')
	with open(log_addr, 'w') as log:
		log.write(f"model: {args.model}, num_trees: {args.num_trees}, learning_rate: {args.learning_rate}\n")
		if args.batch_size > 0:
			log.write(f"batch_size: {args.batch_size}, block_size: {args.block_size}, "
					  f"num_batch: {'full' if args.num_batch is None else args.num_batch}\n")
		log.write("\n")

		for tree_idx in range(args.num_trees):
			log.write(f"[Iter #{tree_idx:4d}] ")
			if "val_losses" in stats.keys():
				log.write(f"val_loss = {stats['val_losses'][tree_idx]:.4f}")
			if "train_losses" in stats.keys():
				log.write(f", train_loss = {stats['train_losses'][tree_idx]:.4f}")
			if "val_criterion" in stats.keys():
				log.write(f", val_obj = {stats['val_criterion'][tree_idx]:.4f}")
			if "train_criterion" in stats.keys():
				log.write(f", train_obj = {stats['train_criterion'][tree_idx]:.4f}")
			log.write("\n")
		log.write(f"Final validation loss: {stats['val_loss']:.4f}")


def create_plot(stats, model_name, save=True):
	"""Create the following plots:
		- Training & validation losses per iteration
		- (if available) Training & validation criterion values per iteration

	:stats: dictionary containing training statistics
	:model_name: name of model given as command argument
		(stored in Argparse parsed object)
	:save: Whether to save the plot with name '{model_name}_curves.png'.
		If False, then views the plot instead.
	:returns: None
	"""
	fig = plt.figure()
	if 'val_losses' in stats.keys():
		ax1 = fig.add_subplot(1,2,1)
		ax1.plot(np.arange(len(stats['val_losses'])) + 1, stats['val_losses'], label='val_loss')
		if 'train_losses' in stats.keys():
			ax1.plot(np.arange(len(stats['train_losses'])) + 1, stats['train_losses'], label='train_loss')
		ax1.set_xlabel('# Trees')
		ax1.set_ylabel('Loss')
		ax1.legend()

	if 'val_criterion' in stats.keys():
		ax2 = fig.add_subplot(1,2,2)
		ax2.plot(np.arange(len(stats['val_criterion'])) + 1, stats['val_criterion'], 'r-', label='val_criterion')
		ax2.set_xlabel('# Trees')
		ax2.set_ylabel('Criterion')
		if 'train_criterion' in stats.keys():
			ax2.plot(np.arange(len(stats['train_criterion'])) + 1, stats['train_criterion'], 'b-', label='train_criterion')
		ax2.legend()

	if save:
		plt.savefig(fname=model_name + "_curves")
	else:
		plt.show()


def gbrt(features, outputs, loss_fn='LSE', lr=1, num_trees=100, verbose=True, is_shuffled=False):
	"""Trains and validates a Scikit-Learn GBRT using the given dataset,
	and reports statistics from training process.

	:features, outputs: the dataset
	:loss_fn: the loss function to quantify the performance of the model
	:lr: the learning rate
	:num_trees: the number of iterations (trees)
	:verbose: prints out the training loss at each iteration (tree)
	:is_shuffled: whether the input dataset has already been shuffled
		prior to this function call
	:returns: A dictionary containing the following:
		- 'val_loss':		The final validation loss
		- 'val_losses':		The validation loss after each iteration
		- 'train_criterion':The score on training set after each iteration
		- 'val_criterion':	The score on validation set after each iteration
	"""
	loss = {'LSE':'ls', 'LAD': 'lad', 'huber':'huber'}[loss_fn]
	params = {
		'n_estimators': num_trees,
		'learning_rate': lr,
		'loss': loss,
		'verbose': verbose,
		}

	f_train, o_train, f_val, o_val = train_test_split(features, outputs, test_size=0.1, shuffle=True)

	model = ensemble.GradientBoostingRegressor(**params)
	model.fit(f_train, o_train)

	val_scores = np.zeros(params['n_estimators'], dtype=np.float64)
	val_losses = np.zeros(params['n_estimators'], dtype=np.float64)

	for it, pred in enumerate(model.staged_predict(f_val)):
		val_scores[it] = model.loss_(o_val, pred)
		val_losses[it] = mean_squared_error(o_val, pred)

	result = {
		'val_loss' 			: val_losses[-1],
		'val_losses'		: val_losses,
		'train_criterion'	: model.train_score_,
		'val_criterion'		: val_scores,
	}
	return result


def xgboost(features, outputs, lr=0.1, num_trees=100, subsample=1, max_depth=1, verbose=True):
	"""Trains and validates a XGBoost GBRT using the given dataset,
	and reports statistics from training process.

	:features, outputs: the dataset
	:lr: the learning rate
	:num_trees: the number of iterations (trees)
	:verbose: prints out the training loss at each iteration (tree)
	:returns: A dictionary containing the following:
		- 'val_loss':		The final validation loss
		- 'val_losses':		The validation loss after each iteration
		- 'train_losses':	The training loss after each iteration
	"""
	f_train, f_val, o_train, o_val = train_test_split(features, outputs, test_size=0.1, shuffle=True)

	params = {
		'n_estimators': num_trees,
		'objective': 'reg:squarederror',
		'learning_rate': lr,
		'verbosity': 2 if verbose else 1,
		'subsample': subsample,
		'max_depth': max_depth,
	}

	model = xgb.XGBRegressor(**params)
	model.fit(f_train, o_train,
			  eval_set = [(f_train,o_train), (f_val, o_val)],
			  eval_metric = 'rmse',
			  verbose=verbose)

	if verbose:
		print(">>> Model training complete")

	evals_result = model.evals_result()

	train_losses = evals_result['validation_0']['rmse'] # 1st arg in `eval_set`
	val_losses   = evals_result['validation_1']['rmse'] # 2nd arg in `eval_set`

	result = {
		'val_loss':		val_losses[-1],
		'val_losses':	val_losses,
		'train_losses':	train_losses,
	}
	return result

def stc_xgboost(features, outputs, lr=0.1, num_trees=100, verbose=True):
	#f_train, f_val, o_train, o_val = train_test_split(features, outputs, test_size=0.1, shuffle=True)

	params = {
		'n_estimators': num_trees,
		'objective': 'reg:squarederror',
		'learning_rate': lr,
		'verbosity': 2 if verbose else 1,
	}

	model = xgb.XGBRegressor(**params)

	subsample = np.linspace(0.1,1,10)
	param_grid = {'subsample': subsample}
	kfold = StratifiedKFold(n_splits=10, shuffle=True)
	grid_search = GridSearchCV(model, param_grid,
							   scoring='neg_mean_squared_error', n_jobs=-1, cv=kfold,
							   verbose=2 if verbose else 1, return_train_score=True)
	search = grid_search.fit(features, outputs)

	return search.cv_results_

def main():
	parsed_args = parser.parse_args()
	conn = create_connection(parsed_args.db_path)

	if parsed_args.verbose:
		start_time = time()

	if parsed_args.model == "gbrt":
		features, outputs = extract_features(conn,
											 table_name = 'rides',
											 variant = 'random' if parsed_args.rand_subset > 0 else 'all',
											 size = parsed_args.rand_subset)
		result = gbrt(features, outputs, verbose=parsed_args.verbose, is_shuffled=(parsed_args.rand_subset > 0))
	elif parsed_args.model == "xgboost":
		features, outputs = extract_features(conn,
											 table_name = 'rides',
											 variant = 'random' if parsed_args.rand_subset > 0 else 'all',
											 size = parsed_args.rand_subset)
		if parsed_args.verbose:
			data_parsed_time = time()
			print(f">>> Data parsing complete, duration: {data_parsed_time - start_time} seconds")
		result = xgboost(features, outputs,
						 lr = parsed_args.learning_rate,
						 num_trees = parsed_args.num_trees,
						 max_depth = parsed_args.max_depth,
						 verbose = parsed_args.verbose,
						 subsample = parsed_args.subsample_rate,
						)
	elif parsed_args.model == "stc_xgboost":
		features, outputs = extract_features(conn,
											 table_name = 'rides',
											 variant = 'random' if parsed_args.rand_subset > 0 else 'all',
											 size = parsed_args.rand_subset)
		if parsed_args.verbose:
			data_parsed_time = time()
			print(f">>> Data parsing complete, duration: {data_parsed_time - start_time} seconds")
		result = stc_xgboost(features, outputs,
							 lr = parsed_args.learning_rate,
							 num_trees = parsed_args.num_trees,
							 verbose = parsed_args.verbose,
							)
		print(result)
		np.save('./cv_result.npy', result)
		conn.close()
		return
	elif parsed_args.model == "lightgbm":
		pass

	if parsed_args.log:
		write_log(args=parsed_args, stats=result)

	if 'val_loss' in result.keys():
		print(f"Validation set MSE = {result['val_loss']}")
	if 'val_losses' in result.keys():
		create_plot(result, parsed_args.model)
		
	conn.close()


if __name__ == "__main__":
	main()
