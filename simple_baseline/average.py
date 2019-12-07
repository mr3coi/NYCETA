import numpy as np
from scipy import sparse
from scipy.stats import linregress
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import argparse
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0],".."))
from obtain_features import *
from baseline_utils import parse_dmat_name, SUPERBORO_CODE

parser = argparse.ArgumentParser(
            description="Uses mean travel time as estimate"
         )

# Method
parser.add_argument("-m", "--method", type=str, choices=["mean","linreg"],
                    help="Method for simple baseline ('mean','linreg')")

# Dataset
parser.add_argument("--db-path", type=str, default="./rides.db",
                    help="Path to the sqlite3 database file.")
parser.add_argument("-sm", "--stddev-mul", type=float,
                    default=1, choices=[-1,0.25,0.5,1,2], help="Number of stddev to add to the cutoff "
                         "for outlier removal. -1 gives the whole dataset "
                         "(choices: -1,0.25,0.5,1,2, default=1, "
                         "assumed to have at most one decimal place)")
parser.add_argument("-doh", "--datetime-one-hot", action="store_true",
                    help="Let the date & time features be loaded as one-hot")
parser.add_argument("-woh", "--weekdays-one-hot", action="store_true",
                    help="Let the week-of-the-day feature be loaded as one-hot")
parser.add_argument("--no-loc-id", dest='loc_id', action="store_false",
                    help="Let the zone IDs be excluded from the dataset")
parser.add_argument("--test-size", type=float, default=0.2,
                    help="Proportion of validation set (default: 0.1)")
parser.add_argument("--start-sb", type=int, default=0, choices=[1,2,3],
                    help="Use subset of data starting at the super-borough "
                         "specified by code; use all data if unspecified "
                         "(1: Bronx, EWR, Manhattan | "
                         "2: Brooklyn, Queens | 3: Staten Island)")
parser.add_argument("--end-sb", type=int, default=0, choices=[1,2,3],
                    help="Use subset of data ending at the  super-borough "
                         "specified by code; use the same value as `--start-sb` "
                         "if not provided and `--start-sb` has been provided, "
                         "else use all data (1: Bronx, EWR, Manhattan | "
                         "2: Brooklyn, Queens | 3: Staten Island)")

# Speeding up
parser.add_argument("--save-path", type=str, default=None,
                    help="Path to saved DMatrix datasets. "
                         "Used to exploit the fast loading speed. "
                         "Only supported when `--method mean`")
parser.add_argument("--use-saved", action="store_true",
                    help="If True, uses saved DMatrix datasets "
                         "the path to which is provided by `--save-path`. "
                         "Will be set automatically when "
                         "`--save-path` is provided")

# Miscellaneous
parser.add_argument("--seed", type=int, default=10701,
                    help="Seed for the whole program")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Let the program be verbose")


def eval_mean(train_targets, test_targets, loss_fn="MSE"):
    loss = {"MSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true,y_pred)),}[loss_fn]
    prediction = np.mean(train_targets)
    return prediction, loss(test_targets, np.ones_like(test_targets) * prediction)


def eval_linreg(f_train, o_train, f_test, o_test, loss_fn="MSE"):
    loss = {"MSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true,y_pred)),}[loss_fn]
    return 0


def main():
    args = parser.parse_args()
    args.use_saved = (args.method == "mean" and args.save_path is not None)

    if args.use_saved:
        test_outputs = xgb.DMatrix(args.save_path + ".val").get_label()
        train_outputs = xgb.DMatrix(args.save_path + ".train").get_label()
        args = parse_dmat_name(args)
    else:
        data_params = {
            "table_name":"rides",
            "variant":"all",
            "datetime_onehot":args.datetime_one_hot,
            "weekdays_onehot":args.weekdays_one_hot,
            "include_loc_ids":args.loc_id,
            "start_super_boro":SUPERBORO_CODE[args.start_sb],
            "end_super_boro":SUPERBORO_CODE[args.end_sb] \
                    if args.end_sb > 0 \
                    else SUPERBORO_CODE[args.start_sb],
            "stddev_multiplier":args.stddev_mul,
        }

        conn = create_connection(args.db_path)
        if args.method == "mean":
            _, outputs = extract_features(conn, **data_params)
            train_outputs, test_outputs = \
                train_test_split(outputs,
                                 test_size=args.test_size,
                                 shuffle=True, random_state=args.seed)
        elif args.method == "linreg":
            features, outputs = extract_features(conn, **data_params)
            train_features, train_outputs, test_features, test_outputs = \
                train_test_split(features, outputs,
                                 test_size=args.test_size,
                                 shuffle=True, random_state=args.seed)

    if args.method == "mean":
        pred, loss = eval_mean(train_outputs, test_outputs)
        print(f">>> Prediction: {pred:.4f}, loss: {loss:.4f}")
    elif args.method == "linreg":
        loss = eval_linreg(train_features, train_outputs,
                           test_features, test_outputs)
        print(f">>> Prediction: loss: {loss:.4f}")


if __name__ == "__main__":
    main()
