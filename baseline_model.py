import numpy as np
from scipy import sparse
import argparse
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import xgboost as xgb

from time import time
import os

from baseline_utils import *
from obtain_features import *


parser = argparse.ArgumentParser(
            description="Configure baseline model and how to train it"
         )

# Model / Training
parser.add_argument("-m", "--model", type=str, default="xgboost",
                    choices = ["gbrt", "xgboost", "xgb_cv",
                               "xgb_gs", "save"],
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
parser.add_argument("--nfold", type=int, default=10,
                    help="The number of folds for k-fold CV")

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
                         "This will be set AUTOMATICALLY as `True` "
                         "once `--save-path` argument is provided")
parser.add_argument("--save-path", type=str, default=None,
                    help="The path to saved DMatrices. Make sure "
                         "to EXCLUDE the extensions")

# Loading & Prediction
parser.add_argument("--models-dir", default="models", type=str,
                    help="The directory containing the trained model "
                         "to predict with (Default: 'models')")
parser.add_argument("--pred-model", default=None, type=str,
                    help="The name of the trained model to predict "
                         "with. The model should be placed in "
                         "the directory specified by `--models-dir`")

# Dataset
parser.add_argument("-sm", "--stddev-mul", type=float,
                    default=1, choices=[-1,0.25,0.5,1,2],
                    help="Number of stddev to add to the cutoff "
                         "for outlier removal. -1 gives the whole dataset "
                         "(choices: -1,0.25,0.5,1,2, default=1, "
                         "assumed to have at most one decimal place)")
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

# Logging / Output
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Let the model print training progress (if supported)")
parser.add_argument("--log", action="store_true",
                    help="Record training log into a text file "
                         "(default location: 'NYCETA/logs')")
parser.add_argument("-sv", "--save-model", action="store_true",
                    help="Save the trained model")


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

    f_train, f_val, o_train, o_val = \
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
            dart_params=None,
            gpu=False,
            n_jobs=4,
            use_saved=False,
            save_path=None,
            verbose=True):
    """Trains an XGBoost GBRT and validates it with a
    holdout, where both training and holdout validation
    datasets are splitted from the given dataset,
    and reports statistics from the training process.

    :features, outputs: The full dataset
    :loss_fn: The loss function to use during training
    :lr: Learning rate for XGBoost
    :booster: Type of booster for each iteration
    :num_trees: The number of iterations (trees)
    :subsample: The subsampling rate of rows
    :max_depth: Maximum depth of each tree booster
    :dart_params: Dictionary containing DART-booster-specific
                  parameters
    :gpu: Whether to use GPU to train or not
    :n_jobs: When not using GPU, number of threads to use to train
    :use_saved: Whether to use data already preprocessed and
                saved to disk in the form of `xgb.DMatrix`
                (run with `--model save` option first)
    :save_path: The path to the stored `xgb.DMatrix`
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
            assert dart_params is not None, \
                "ERROR: Please provide DART-related parameters"
            params['rate_drop'] = dart_params['rate_drop']
            params['sample_type'] = dart_params['sample_type']
            params['normalize_type'] = dart_params['normalize_type']

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
            assert dart_params is not None, \
                "ERROR: Please provide DART-related parameters"
            params['rate_drop'] = dart_params['rate_drop']
            params['sample_type'] = dart_params['sample_type']
            params['normalize_type'] = dart_params['normalize_type']

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
                          early_stopping_rounds=10,
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
    return result, model


def xgb_cv(features=None, outputs=None,
           nfold=10,
           loss_fn="LSE",
           lr=0.1,
           num_trees=100,
           booster="gbtree",
           subsample=1,
           max_depth=3,
           dart_params=None,
           gpu=False,
           use_saved=False,
           save_path=None,
           seed=None,
           verbose=False):
    """Trains and conducts CV over an XGBoost GBRT,
    and reports statistics from the training process.
    No separate validation dataset is provided.
    Refer to `xgboost` for parameters not described below.

    :features, outputs: The FULL dataset
    :nfold: Number of folds for K-fold CV
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
        dtrain = xgb.DMatrix(features, label=outputs)

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
            assert dart_params is not None, \
                "ERROR: Please provide DART-related parameters"
            params['rate_drop'] = dart_params['rate_drop']
            params['sample_type'] = dart_params['sample_type']
            params['normalize_type'] = dart_params['normalize_type']

        cv_result = xgb.cv(params, dtrain,
                           num_boost_round=num_trees,
                           nfold=nfold,
                           metrics=[loss],
                           shuffle=True,
                           seed=seed,
                           early_stopping_rounds=10,
                           verbose_eval=verbose,
                           show_stdv=verbose,
                           as_pandas=False,
                          )

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
            assert dart_params is not None, \
                "ERROR: Please provide DART-related parameters"
            params["rate_drop"] = dart_params["rate_drop"]
            params["sample_type"] = dart_params["sample_type"]
            params["normalize_type"] = dart_params["normalize_type"]

        dtrain = xgb.DMatrix(save_path + ".train")

        cv_result = xgb.cv(params, dtrain,
                           num_boost_round=num_trees,
                           nfold=nfold,
                           metrics=[loss],
                           shuffle=True,
                           seed=seed,
                           early_stopping_rounds=10,
                           verbose_eval=verbose,
                           show_stdv=verbose,
                           as_pandas=False,
                          )

    if verbose:
        print(">>> Model training complete")

    train_losses = cv_result[f"train-{loss}-mean"]
    val_losses   = cv_result[f"test-{loss}-mean"]
    train_losses_std = cv_result[f"train-{loss}-std"]
    val_losses_std   = cv_result[f"test-{loss}-std"]
    result = {}
    return result


def xgb_gridsearch(features, outputs,
                   param_grid,
                   nfold=10,
                   loss_fn="LSE",
                   lr=0.1,
                   num_trees=100,
                   booster="gbtree",
                   subsample=1,
                   max_depth=3,
                   gpu=False,
                   n_jobs=4,
                   verbose=True):
    """Conducts K-fold CV for grid search on
    hyperparameters of XGBoost.
    NOTE) This does NOT support `--use-saved' option
    and ONLY supports passing in data as np.arrays
    or sparse arrays. Hence, this may suffer from
    memory issues depending on input size. Consider
    implementing np.array version of `save_dmatrix`
    function in `baseline_utils.py`.
    Refer to `xgboost` for parameters not described below.

    :features, outputs: The FULL dataset
    :nfold: Number of splits for K-fold CV
    :param_grid: Dictionary where each entry is
        - key: Name of parameter
               (refer to 'xgb.XGBRegressor' for details)
        - value: A list or 1-D array of values to try
    :returns: A dictionary containing collected statistics.
        Refer to 'sklearn.model_selection.GridSearchCV' doc
        for details.
    """
    loss = {"LSE": "neg_mean_squared_error"}[loss_fn]
    objective = {"LSE": "reg:squarederror"}[loss_fn]

    params = {
        "n_estimators": num_trees,
        "objective": objective,
        "learning_rate": lr,
        "verbosity": 2 if verbose else 1,
        "tree_method": "gpu_hist" if gpu else "approx",
        "booster": booster,
        "subsample": subsample,
        "max_depth": max_depth,
    }
    if not gpu:
        params['n_jobs'] = n_jobs

    model = xgb.XGBRegressor(**params)

    kfold = StratifiedKFold(n_splits=nfold, shuffle=True)
    grid_search = GridSearchCV(model, param_grid,
                               scoring=loss,
                               n_jobs=-1,
                               cv=kfold,
                               verbose=2 if verbose else 1,
                               return_train_score=True)
    search = grid_search.fit(features, outputs)

    return search.cv_results_


def xgb_load_and_predict(model_path, dmat_path, loss_fn="MSE"):
    """Loads a stored model and evaluates it against a
    validation dataset stored in the form of a DMatrix.
    Mainly intended to evaluate large models (baseline or
    ensemble models) within a subset of superboroughs

    :model_path: Path to the saved model
    :dmat_path: Path to the stored validation dmatrix to
                evaluate against (EXCLUDING the suffix)
    :loss_fn: Type of loss function to evaluate predictions with.
              Currently supported: ["MSE"]
    :returns: Loss value frome evaluation
    """
    loss = {"MSE": lambda y,fx: np.sqrt(mean_squared_error(y,fx))}[loss_fn]

    model = xgb.Booster(model_file=model_path)
    val_data = xgb.DMatrix(dmat_path + ".val")
    predictions = model.predict(val_data)
    targets = val_data.get_label()
    return loss(targets,predictions)


def main():
    parsed_args = parser.parse_args()

    if parsed_args.pred_model is not None:
        model_path = os.path.join(parsed_args.models_dir,
                                  parsed_args.pred_model)
        loss = xgb_load_and_predict(model_path, parsed_args.save_path)
        print(f">>> Loss of prediction : {loss}")
        return

    conn = create_connection(parsed_args.db_path)

    # Set `--use-saved` automatically if `--save-path` has been given
    parsed_args.use_saved = (parsed_args.save_path is not None)

    if not parsed_args.use_saved:
        data_params = {
            "table_name":"rides",
            "variant":"random" if parsed_args.rand_subset > 0 else "all",
            "size":parsed_args.rand_subset,
            "datetime_onehot":parsed_args.datetime_one_hot,
            "weekdays_onehot":parsed_args.weekdays_one_hot,
            "include_loc_ids":parsed_args.loc_id,
            "start_super_boro":SUPERBORO_CODE[parsed_args.start_sb],
            "end_super_boro":SUPERBORO_CODE[parsed_args.end_sb] \
                    if parsed_args.end_sb > 0 \
                    else SUPERBORO_CODE[parsed_args.start_sb],
            "stddev_multiplier":parsed_args.stddev_mul,
        }
        if parsed_args.verbose:
            start_time = time()

        features, outputs = \
            extract_features(conn, **data_params)
    else:
        # Parse dataset configurations from stored name
        parsed_args = parse_dmat_name(parsed_args)

    dart_params = {
         "rate_drop":parsed_args.rate_drop,
         "sample_type":parsed_args.sample_type,
         "normalize_type":parsed_args.normalize_type,
    } if parsed_args.booster == "dart" else None

    xgb_params = {
         "booster":parsed_args.booster,
         "lr":parsed_args.learning_rate,
         "num_trees":parsed_args.num_trees,
         "max_depth":parsed_args.max_depth,
         "subsample":parsed_args.subsample_rate,
         "gpu":parsed_args.gpu,
         "dart_params":dart_params,
         "verbose":parsed_args.verbose,
    }

    # `xgb_gs` doesn't support `--use-saved`; see docs
    if parsed_args.model != "xgb_gs":
         xgb_params["use_saved"] = parsed_args.use_saved
         xgb_params["save_path"] = parsed_args.save_path

    if parsed_args.model == "gbrt":
        result = gbrt(features, outputs, verbose=parsed_args.verbose)

    elif parsed_args.model == "xgboost":
        if not parsed_args.use_saved and parsed_args.verbose:
            data_parsed_time = time()
            print(">>> Data parsing complete, "
                  f"duration: {data_parsed_time - start_time} seconds")
        xgb_params["n_jobs"] = parsed_args.xgb_num_thread
        result, model = xgboost(features if not parsed_args.use_saved else None,
                                outputs  if not parsed_args.use_saved else None,
                                **xgb_params,
                               )

    elif parsed_args.model == "xgb_gs":
        if parsed_args.verbose:
            data_parsed_time = time()
            print(">>> Data parsing complete, "
                  f"duration: {data_parsed_time - start_time} seconds")

        # Set up parameter values to do grid-search over here
        param_grid = {"subsample": np.linspace(0.1,1,10)}

        result = xgb_gridsearch(features, outputs,
                                param_grid=param_grid,
                                nfold=parsed_args.nfold,
                                **xgb_params,)
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
        np.save("./gs_result.npy", result)
        conn.close()
        return

    elif parsed_args.model == "xgb_cv":
        if not parsed_args.use_saved and parsed_args.verbose:
            data_parsed_time = time()
            print(">>> Data parsing complete, "
                  f"duration: {data_parsed_time - start_time} seconds")
        result = xgb_cv(features if not parsed_args.use_saved else None,
                        outputs  if not parsed_args.use_saved else None,
                        nfold=parsed_args.nfold,
                        seed=10701,
                        **xgb_params,
                       )
    elif parsed_args.model == "save":
        save_dmatrix(features, outputs, parsed_args, seed=10701)
        return

    if parsed_args.log:
        log_time = write_log(args=parsed_args, stats=result)

    if "val_loss" in result.keys():
        print(f"Validation set MSE = {result['val_loss']}")
    if "val_losses" in result.keys():
        create_plot(result, parsed_args.model)

    if parsed_args.save_model:
        assert parsed_args.model == "xgboost", \
            "ERROR: only 'xgboost' model can be saved"
        model_path = xgb_save_model(model, log_time, parsed_args)
        if parsed_args.verbose:
            print(f">>> Model saved as: {model_path}")
        
    conn.close()


if __name__ == "__main__":
    main()
