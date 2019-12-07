import numpy as np
from scipy import sparse
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import argparse
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0],".."))
from obtain_features import *

parser = argparse.ArgumentParser(
            description="Uses mean travel time as estimate"
         )

# Dataset
parser.add_argument("--db-path", type=str, default="../rides.db",
                    help="Path to SQL DB file with raw data")
parser.add_argument("--save-path", type=str, default=None,
                    help="Path to saved DMatrix datasets. "
                         "Used to exploit the fast loading speed")
parser.add_argument("--use-saved", action="store_true",
                    help="If True, uses saved DMatrix datasets "
                         "the path to which is provided by `--save-path`. "
                         "Will be set automatically when "
                         "`--save-path` is provided")
parser.add_argument("--test-size", type=float, default=0.2,
                    help="Proportion of test set wrt "
                         "the whole raw dataset")
parser.add_argument("--stddev-mul", type=float, default=1.0,
                    help="Number of stddevs to add to cutoff "
                         "for preprocessing the dataset")

# Miscellaneous
parser.add_argument("--seed", type=int, default=10701,
                    help="Seed for the whole program")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Let the program be verbose")


def evaluate(train_targets, test_targets, loss_fn="MSE"):
    loss = {"MSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true,y_pred)),}[loss_fn]
    prediction = np.mean(train_targets)
    return prediction, loss(test_targets, np.ones_like(test_targets) * prediction)


def main():
    args = parser.parse_args()
    args.use_saved = (args.save_path is not None)

    if args.use_saved:
        test_outputs = xgb.DMatrix(args.save_path + ".val").get_label()
        train_outputs = xgb.DMatrix(args.save_path + ".train").get_label()
    else:
        conn = create_connection(parsed_args.db_path)
        _, outputs = extract_features(conn,
                                      table_name = "rides",
                                      variant = "all",)
        train_outputs, test_outputs = \
            train_test_split(outputs, test_size=args.test_size,
                             shuffle=True, random_state=args.seed)

    pred, loss = evaluate(train_outputs, test_outputs)
    print(f">>> Prediction: {pred:.4f}, loss: {loss:.4f}")


if __name__ == "__main__":
    main()
