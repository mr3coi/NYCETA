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


parser = argparse.ArgumentParser(
            description="Configure baseline model and how to train it"
         )

# Model / Training
parser.add_argument("-m", "--model", type=str, default="xgboost",
                    choices = ["gbrt", "xgboost", "lightgbm", "xgboost_cv", "save"],
                    help="Choose which baseline model to train "
                         "(default: xgboost)")
parser.add_argument("-b", "--booster", type=str, default="gbtree",
                    choices = ["gbtree", "gblinear", "dart"],
                    help="Choose which weak learner to use for XGBoost "
                         "(default: gbtree)")
parser.add_argument("--num-trees", type=int, default=100,
                    help="Number of trees (iterations) to train")
parser.add_argument("--max-depth", type=int, default=3,
                    help="The maximum depth of each regression tree")
parser.add_argument("-lr", "--learning-rate", type=float, default=0.1,
                    help="The learning rate to train the model with")
parser.add_argument("-ssr", "--subsample-rate", type=float, default=1,
                    help="Subsampling rate for rows. "
                         "Must be between 0 and 1.")

# DART-specific
parser.add_argument("--rate-drop", type=float, default=0.1,
                    help="Dropout rate for DART. "
                         "Must first set '--booster dart' option")
parser.add_argument("-st", "--sample-type", type=str, default="uniform",
                    choices=["uniform","weighted"],
                    help="Sampling algo for DART. "
                         "Must first set '--booster dart' option")
parser.add_argument("-nt", "--normalize-type", type=str, default="tree",
                    choices=["tree","forest"],
                    help="Normalization algo for DART. "
                         "Must first set '--booster dart' option")

# Speeding-up Training
parser.add_argument("--gpu", action="store_true",
                    help="Let the model train on GPU. "
                         "Currently only supported by 'xgboost'")
parser.add_argument("--xgb-num-thread", type=int, default=4,
                    help="Number of parallel threads for XGBoost")
parser.add_argument("--use-saved", action="store_true",
                    help="Use the preprocessed & saved DMatrix data. "
                         "Need to first run with '--model save'. "
                         "Also, recommend passing the same options "
                         "as was used provided in '--model save' call "
                         "for logging purposes")
parser.add_argument("--save-path", type=str, default=None,
                    help="The path to saved DMatrices. Make sure "
                         "to EXCLUDE the extensions")

# Dataset
parser.add_argument("--db-path", type=str, default="./rides.db",
                    help="Path to the sqlite3 database file.")
parser.add_argument("-r", "--rand-subset", type=int, default=0,
                    help="Shuffle the dataset, then sample a subset "
                         "with size specified by argument (default: 0). "
                         "Size 0 means the whole dataset is used (i.e. variant='all')")
parser.add_argument("--batch-size", type=int, default=-1,
                    help="Batch size for semi-random batch learning")
parser.add_argument("-doh", "--datetime-one-hot", action="store_true",
                    help="Let the date & time features be loaded as one-hot")
parser.add_argument("-woh", "--weekdays-one-hot", action="store_true",
                    help="Let the week-of-the-day feature be loaded as one-hot")
parser.add_argument("--no-loc-id", dest='loc_id', action="store_false",
                    help="Let the zone IDs be excluded from the dataset")
parser.add_argument("--test-size", type=float, default=0.1,
                    help="Proportion of validation set (default: 0.1)")

# Logging / Output
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Let the model print training progress (if supported)")
parser.add_argument("--log", action="store_true",
                    help="Record training log into a text file "
                         "(default location: 'NYCETA/logs')")


def create_dir(dirname):
    """Creates a directory in project root directory
    to store training logs, unless already exists.

    :dirname: the name of the directory
    :returns: the path to the directory
    """
    dir_path = os.path.join(os.getcwd(),dirname)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def write_log(args, stats, dirname="logs"):
    """Store a log of losses per each iteration onto a file.
    Creates a directory under current pwd named {dirname},
    and stores hyperparameters as well as loss/objective values
    depending on which are available in 'stats' dictionary.
    Each log is named after the current time of running the code.

    :args: Argparse parsed object 
    :stats: dictionary containing training statistics
    :dirname: name of the directory to contain log files
    :returns: None
    """

    curr_time = dt.now().astimezone(timezone("US/Eastern")) \
                        .strftime("%Y-%m-%d-%H-%M-%S")
    log_addr = os.path.join(create_dir(dirname), f"log_{curr_time}.txt")
    with open(log_addr, "w") as log:
        log.write(f"model: {args.model}, num_trees: {args.num_trees}, "
                  f"max_depth: {args.max_depth}, "
                  f"booster: {args.booster}\n")
        '''XXX: Deprecated
        if args.batch_size > 0:
            log.write(f"batch_size: {args.batch_size}, "
                      f"block_size: {args.block_size}, "
                      f"num_batch: {'full' if args.num_batch is None else args.num_batch}\n")
        '''
        log.write(f"subsample_rate: {args.subsample_rate}, "
                  f"learning_rate: {args.learning_rate}\n")
        log.write(f"datetime_one_hot: {args.datetime_one_hot}, "
                  f"weekdays_one_hot: {args.weekdays_one_hot}, "
                  f"loc_id: {args.loc_id}, "
                  f"test_size: {args.test_size}")
        log.write("\n\n")

        for tree_idx in range(args.num_trees):
            log.write(f"[Iter #{tree_idx+1:4d}] ")
            if "val_losses" in stats.keys():
                log.write(f"val_loss = {stats['val_losses'][tree_idx]:.4f}")
            if "train_losses" in stats.keys():
                log.write(f", train_loss = {stats['train_losses'][tree_idx]:.4f}")
            if "val_objective" in stats.keys():
                log.write(f", val_obj = {stats['val_objective'][tree_idx]:.4f}")
            if "train_objective" in stats.keys():
                log.write(f", train_obj = {stats['train_objective'][tree_idx]:.4f}")
            log.write("\n")
        log.write(f"Final validation loss: {stats['val_loss']:.4f}")


def create_plot(stats, model_name, save=True):
    """Create the following plots (if available):
        - Training & validation losses per iteration
        - Training & validation objective values per iteration

    :stats: dictionary containing training statistics
    :model_name: name of model given as command argument
        (stored in Argparse parsed object)
    :save: Whether to save the plot with name '{model_name}_curves.png'.
        If False, then views the plot instead.
    :returns: None
    """
    fig = plt.figure()
    if "val_losses" in stats.keys():
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(np.arange(len(stats["val_losses"])) + 1,
                 stats["val_losses"],
                 label="val_loss")
        if "train_losses" in stats.keys():
            ax1.plot(np.arange(len(stats["train_losses"])) + 1,
                     stats["train_losses"],
                     label="train_loss")
        ax1.set_xlabel("# Trees")
        ax1.set_ylabel("Loss")
        ax1.legend()

    if "val_objective" in stats.keys():
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(np.arange(len(stats["val_objective"])) + 1,
                 stats["val_objective"],
                 "r-",
                 label="val_objective")
        ax2.set_xlabel("# Trees")
        ax2.set_ylabel("Criterion")
        if "train_objective" in stats.keys():
            ax2.plot(np.arange(len(stats["train_objective"])) + 1,
                     stats["train_objective"],
                     "b-",
                     label="train_objective")
        ax2.legend()

    if save:
        plt.savefig(fname=model_name + "_curves")
    else:
        plt.show()


def gbrt(features, outputs,
         loss_fn="LSE",
         lr=1,
         num_trees=100,
         verbose=True):
    """Trains and validates a Scikit-Learn GBRT using the given dataset,
    and reports statistics from training process.

    :features, outputs: The dataset
    :loss_fn: The loss function to use during training
    :lr: Learning rate for GBRT
    :num_trees: The number of iterations (trees)
    :verbose: Prints out the training loss at each iteration (tree)
    :returns: A dictionary containing the following:
        - 'val_loss':       The final validation loss
        - 'val_losses':     The validation loss after each iteration
        - 'train_objective':The score on training set after each iteration
        - 'val_objective':  The score on validation set after each iteration
    """
    loss = {"LSE":"ls", "LAD": "lad", "HUBER":"huber"}[loss_fn]
    params = {
        "n_estimators": num_trees,
        "learning_rate": lr,
        "loss": loss,
        "verbose": verbose,
        }

    f_train, o_train, f_val, o_val = \
        train_test_split(features, outputs,
                         test_size=0.1,
                         shuffle=True,
                         random_state=10701)

    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(f_train, o_train)

    val_scores = np.zeros(params["n_estimators"], dtype=np.float64)
    val_losses = np.zeros(params["n_estimators"], dtype=np.float64)

    for it, pred in enumerate(model.staged_predict(f_val)):
        val_scores[it] = model.loss_(o_val, pred)
        val_losses[it] = mean_squared_error(o_val, pred)

    result = {
        "val_loss"          : val_losses[-1],
        "val_losses"        : val_losses,
        "train_objective"   : model.train_score_,
        "val_objective"     : val_scores,
    }
    return result


def xgboost(features=None, outputs=None,
            loss_fn="LSE",
            lr=0.1,
            num_trees=100,
            booster="gbtree",
            subsample=1,
            max_depth=3,
            rate_drop=None,
            sample_type=None,
            normalize_type=None,
            gpu=False,
            n_jobs=4,
            use_saved=False,
            save_path=None,
            verbose=True):
    """Trains and validates a XGBoost GBRT using the given dataset,
    and reports statistics from training process.

    :features, outputs: The full dataset
    :loss_fn: The loss function to use during training
    :lr: Learning rate for XGBoost
    :num_trees: The number of iterations (trees)
    :verbose: Prints out the training loss at each iteration (tree)
    :returns: A dictionary containing the following:
        - "val_loss":       The final validation loss
        - "val_losses":     The validation loss after each iteration
        - "train_losses":   The training loss after each iteration
    """
    loss = {"LSE": "rmse"}[loss_fn]
    objective = {"LSE": "reg:squarederror"}[loss_fn]

    if not use_saved:
        assert features is not None and outputs is not None, \
            "ERROR: Please provide `features` or `outputs`."

        f_train, f_val, o_train, o_val = \
            train_test_split(features, outputs,
                             test_size=0.1,
                             shuffle=True,
                             random_state=10701)

        params = {
            "tree_method": "gpu_hist" if gpu else "approx",
            "n_estimators": num_trees,
            "booster": booster,
            "objective": objective,
            "learning_rate": lr,
            "verbosity": 2 if verbose else 1,
            "subsample": subsample,
            "max_depth": max_depth,
        }
        if not gpu:
            params['n_jobs'] = n_jobs
        if booster == "dart":
            params['rate_drop'] = rate_drop
            params['sample_type'] = sample_type
            params['normalize_type'] = normalize_type

        model = xgb.XGBRegressor(**params)
        model.fit(f_train, o_train,
                  eval_set = [(f_train,o_train), (f_val, o_val)],
                  eval_metric = loss,
                  verbose=verbose)

        evals_result = model.evals_result()

        train_losses = evals_result["validation_0"][loss] # 1st arg in `eval_set`
        val_losses   = evals_result["validation_1"][loss] # 2nd arg in `eval_set`
    else:
        assert save_path is not None, \
            "ERROR: Need to provide 'save_path'."
        params = {
            "tree_method": "gpu_hist" if gpu else "approx",
            "booster": booster,
            "objective": objective,
            "learning_rate": lr,
            "verbosity": 2 if verbose else 1,
            "subsample": subsample,
            "max_depth": max_depth,
            "eval_metric": loss,
        }
        if booster == "dart":
            params['rate_drop'] = rate_drop
            params['sample_type'] = sample_type
            params['normalize_type'] = normalize_type

        dtrain = xgb.DMatrix(save_path + '.train')
        dval = xgb.DMatrix(save_path + '.val')
        watchlist = [(dtrain,"train"),(dval,"validation")]
        evals_result = {}

        model = xgb.train(params,
                          dtrain=dtrain,
                          num_boost_round=num_trees,
                          evals=watchlist,
                          verbose_eval=verbose,
                          evals_result=evals_result,
                         )

        train_losses = evals_result["train"][loss] # 1st arg in `eval_set`
        val_losses   = evals_result["validation"][loss] # 2nd arg in `eval_set`

    if verbose:
        print(">>> Model training complete")
    result = {
        "val_loss":     val_losses[-1],
        "val_losses":   val_losses,
        "train_losses": train_losses,
    }
    return result


def xgboost_cv(features, outputs,
               param_grid,
               n_splits=10,
               loss_fn="LSE",
               lr=0.1,
               num_trees=100,
               verbose=True):
    """Conducts K-fold CV for grid search on
    hyperparameters of XGBoost.

    :features, outputs: The full dataset
    :n_splits: Number of splits for K-fold CV
    :param_grid: Dictionary where each entry is
        - key: Name of parameter
               (refer to 'xgb.XGBRegressor' for details)
        - value: A list or 1-D array of values to try
    :loss_fn: The loss function to use during training
    :lr: Learning rate for XGBoost
    :num_trees: The number of iterations (trees) for XGBoost
    :verbose: Prints out the training loss at each iteration (tree)
    :returns: A dictionary containing collected statistics.
        Refer to 'sklearn.model_selection.GridSearchCV'
        for details.
    """
    loss = {"LSE": "neg_mean_squared_error"}[loss_fn]
    objective = {"LSE": "reg:squarederror"}[loss_fn]

    params = {
        "n_estimators": num_trees,
        "objective": objective,
        "learning_rate": lr,
        "verbosity": 2 if verbose else 1,
    }

    model = xgb.XGBRegressor(**params)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    grid_search = GridSearchCV(model, param_grid,
                               scoring=loss,
                               n_jobs=-1,
                               cv=kfold,
                               verbose=2 if verbose else 1,
                               return_train_score=True)
    search = grid_search.fit(features, outputs)

    return search.cv_results_


def save_dmatrix(features, outputs, args, seed=None):
    """Save training/validation DMatrices for XGBoost
    under configurations given by `args`.
    Configurable items are (refer to parser for details):
    - datetime-one-hot
    - weekdays-one-hot
    - no-loc-id
    - test-size
    The DMatrices are stored in 'data' dir under project
    root, and their names encode the configurations.

    :features, outputs: Loaded datasets (NumPy arrays)
    :args: Argparse object
    :seed: Seed for randomizing `train_test_split`
    :returns: None
    """
    # Record configurations in dataset name
    save_name = "dm"
    if args.datetime_one_hot:
        save_name += "_doh"
    if args.weekdays_one_hot:
        save_name += "_woh"
    if args.loc_id:
        save_name += "_locid"
    save_name += f"_s{seed}" if seed is not None else "_random"
    save_name += f"_test{args.test_size}"

    data_dirpath = create_dir("data")
    train_path = os.path.join(data_dirpath, save_name + '.train')
    val_path = os.path.join(data_dirpath, save_name + '.val')

    # Split data with specified `test_size`
    f_train, f_val, o_train, o_val = \
        train_test_split(features, outputs,
                         test_size=args.test_size,
                         shuffle=True,
                         random_state=seed,)

    # Store DMatrices
    dtrain = xgb.DMatrix(f_train, label=o_train)
    dval = xgb.DMatrix(f_val, label=o_val)
    if args.verbose:
        print(">>> Conversion to DMatrix complete")
    dtrain.save_binary(train_path)
    dval.save_binary(val_path)
    if args.verbose:
        print(">>> DMatrices saved to disk")


def main():
    parsed_args = parser.parse_args()
    conn = create_connection(parsed_args.db_path)

    if not parsed_args.use_saved:
        if parsed_args.verbose:
            start_time = time()
        features, outputs = \
            extract_features(conn,
                             table_name="rides",
                             variant="random" if parsed_args.rand_subset > 0 else "all",
                             size=parsed_args.rand_subset,
                             datetime_onehot=parsed_args.datetime_one_hot,
                             weekdays_onehot=parsed_args.weekdays_one_hot,
                             include_loc_ids=parsed_args.loc_id,
                            )

    if parsed_args.model == "gbrt":
        result = gbrt(features, outputs, verbose=parsed_args.verbose)
    elif parsed_args.model == "xgboost":
        if not parsed_args.use_saved and parsed_args.verbose:
            data_parsed_time = time()
            print(">>> Data parsing complete, "
                  f"duration: {data_parsed_time - start_time} seconds")
        params = {
             "booster":parsed_args.booster,
             "lr":parsed_args.learning_rate,
             "num_trees":parsed_args.num_trees,
             "max_depth":parsed_args.max_depth,
             "verbose":parsed_args.verbose,
             "subsample":parsed_args.subsample_rate,
             "gpu":parsed_args.gpu,
             "n_jobs":parsed_args.xgb_num_thread,
             "use_saved":parsed_args.use_saved,
             "save_path":parsed_args.save_path if parsed_args.use_saved else None,
             "rate_drop":parsed_args.rate_drop,
             "sample_type":parsed_args.sample_type,
             "normalize_type":parsed_args.normalize_type,
        }
        result = xgboost(features if not parsed_args.use_saved else None,
                         outputs  if not parsed_args.use_saved else None,
                         **params,
                        )
    elif parsed_args.model == "xgboost_cv":
        if parsed_args.verbose:
            data_parsed_time = time()
            print(">>> Data parsing complete, "
                  f"duration: {data_parsed_time - start_time} seconds")

        # Set up parameter values to do grid-search over here
        param_grid = {"subsample": np.linspace(0.1,1,10)}

        result = xgboost_cv(features, outputs,
                            param_grid = param_grid,
                            lr = parsed_args.learning_rate,
                            num_trees = parsed_args.num_trees,
                            verbose = parsed_args.verbose,
                           )
        if parsed_args.verbose:
            print(result)

        """
        NOTE: Outputs a totally different set of statistics
              compared to other models.
              Since the output is much more comprehensive,
              it has been stored as file for analysis.
              Refer to 'GridSearchCV' in SKLearn for details
              on the output object.
        """
        np.save("./cv_result.npy", result)
        conn.close()
        return
    elif parsed_args.model == "lightgbm":
        pass
    elif parsed_args.model == "save":
        save_dmatrix(features, outputs, parsed_args, seed=10701)
        return

    if parsed_args.log:
        write_log(args=parsed_args, stats=result)

    if "val_loss" in result.keys():
        print(f"Validation set MSE = {result['val_loss']}")
    if "val_losses" in result.keys():
        create_plot(result, parsed_args.model)
        
    conn.close()


if __name__ == "__main__":
    main()
