import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from datetime import datetime as dt
from pytz import timezone
import os

SUPERBORO_CODE = {
    0: None,
    1: ["Bronx", "EWR", "Manhattan"],
    2: ["Brooklyn", "Queens"],
    3: ["Staten Island"],
}

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
        log.write(f"subsample_rate: {args.subsample_rate}, "
                  f"learning_rate: {args.learning_rate}\n")
        log.write(f"datetime_one_hot: {args.datetime_one_hot}, "
                  f"weekdays_one_hot: {args.weekdays_one_hot}, "
                  f"loc_id: {args.loc_id}, "
                  f"stddev_mul: {args.stddev_mul:.1}, "
                  f"test_size: {args.test_size}\n")
        log.write(f"start_sb: {args.start_sb}, ")
        log.write(f"end_sb: {args.end_sb}")
        log.write("\n\n")

        for tree_idx in range(args.num_trees):
            try:
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
            except IndexError:
                break
        log.write(f"Final validation loss: {stats['val_loss']:.4f}")
    return curr_time


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
    save_name += f"_sb{args.start_sb}"
    save_name += f"{args.end_sb}" if args.end_sb > 0 \
                 else f"{args.start_sb}"
    save_name += f"_sm{args.stddev_mul:.1}"
    save_name += f"_test{args.test_size}"
    save_name += f"_doh{int(args.datetime_one_hot)}"
    save_name += f"_woh{int(args.weekdays_one_hot)}"
    save_name += f"_locid{int(args.loc_id)}"
    save_name += f"_s{seed}" if seed is not None else "_random"

    data_dirpath = create_dir("data")
    train_path = os.path.join(data_dirpath, save_name + '.train')
    val_path = os.path.join(data_dirpath, save_name + '.val')

    # Split data with specified `test_size`
    if args.test_size > 0:
        f_train, f_val, o_train, o_val = \
            train_test_split(features, outputs,
                             test_size=args.test_size,
                             shuffle=True,
                             random_state=seed,)
    else:
        f_train, o_train = features, outputs

    # Store DMatrices
    dtrain = xgb.DMatrix(f_train, label=o_train)
    if args.test_size > 0:
        dval = xgb.DMatrix(f_val, label=o_val)
    if args.verbose:
        print(">>> Conversion to DMatrix complete")
    dtrain.save_binary(train_path)
    if args.test_size > 0:
        dval.save_binary(val_path)
    if args.verbose:
        print(">>> DMatrices saved to disk")

def parse_dmat_name(args):
    """Parses information from the name of the stored
    DMatrix dataset, and updates `args` to contain
    the parsed information

    :args: Argparse ArgumentParser object containing
           parsed details. Dataset name string is one
           of the items within `args`
    :returns: `args` with dataset-related items updated
              according to info from dataset name
    """
    dmat_name = args.save_path.split("/")[-1]
    items = dmat_name.split("_")[1:]

    # NOTE) prefixes = ["sb","sm","test","doh","woh","locid","seed"]
    args.start_sb = int(items[0][2])
    args.end_sb = int(items[0][2+1])
    args.stddev_mul = float(items[1][2:])
    args.test_size = float(items[2][4:])
    args.datetime_one_hot = bool(int(items[3][3]))
    args.weekdays_one_hot = bool(int(items[4][3]))
    args.loc_id = bool(int(items[5][5]))

    return args

def xgb_save_model(model, train_time_str, args):
    """Save trained XGBoost model into directory
    specified by `--models-dir` argument

    :model: The trained XGBoost model
    :train_time_str: String containing time that the log
        for training the model has been created
        (to allow for matching with training log)
    :args: Argparse ArgumentParser object
    :returns: Path to the saved model
    """
    models_dir = create_dir(args.models_dir)
    model_name = f"sb{args.start_sb}{args.end_sb}_sm{args.stddev_mul:.1}" \
                 f"_{int(args.datetime_one_hot)}" \
                 f"{int(args.weekdays_one_hot)}" \
                 f"{int(args.loc_id)}"
    model_name += f"_{train_time_str}.xgb"
    model_path = os.path.join(models_dir,model_name)
    model.save_model(model_path)
    return model_path
