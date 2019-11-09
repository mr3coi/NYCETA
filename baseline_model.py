import numpy as np
from scipy import sparse
import argparse

from obtain_features import *

parser = argparse.ArgumentParser(description="Specify baseline model to train")
parser.add_argument('-m', "--model", type=str, choices = ["gbrt", "xgboost", "lightgbm"], default="gbrt",
					help="Choose which baseline model to train")
parser.add_argument('--db-path', type=str, default="./rides.db",
					help="Path to the sqlite3 database file.")
parser.add_argument('-r', '--rand-subset', type=int, default=0,
					help="Shuffle the dataset, then sample a subset with size specified by argument.")

def main():
	parsed_args = parser.parse_args()
	conn = create_connection(parsed_args.db_path)
	features, outputs = extract_features(conn,
										 table_name = 'rides',
										 variant = 'random' if parsed_args.rand_subset > 0 else 'all',
										 random_size = parsed_args.rand_subset)

	if parsed_args.model is "gbrt":
		pass
	elif parsed_args.model is "xgboost":
		pass
	elif parsed_args.model is "lightgbm":
		pass

	conn.close()

if __name__ == "__main__":
	main()
