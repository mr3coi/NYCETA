import xgboost as xgb
import numpy as np
from scipy import sparse

import os
import argparse
from time import time

from baseline_utils import parse_dmat_name, SUPERBORO_CODE
from obtain_features import *


parser = argparse.ArgumentParser(
    description="Integrates trained XGBoost models to "
        "do inference on cross-superboro data points."
    )

# Super-boro models for inference
# TODO: Fill in default values
parser.add_argument("--sb1-model-path", type=str, default="",
                    help="Path to the stored model for Super-boro 1 (MEBx)")
parser.add_argument("--sb2-model-path", type=str, default="",
                    help="Path to the stored model for Super-boro 2 (BkQ)")
parser.add_argument("--sb3-model-path", type=str, default="",
                    help="Path to the stored model for Super-boro 3 (St)")

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
parser.add_argument("-doh", "--datetime-one-hot", action="store_true",
                    help="Let the date & time features be loaded as one-hot")
parser.add_argument("-woh", "--weekdays-one-hot", action="store_true",
                    help="Let the week-of-the-day feature be loaded as one-hot")
parser.add_argument("--no-loc-id", dest='loc_id', action="store_false",
                    help="Let the zone IDs be excluded from the dataset")
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

parser.add_argument("--use-saved", action="store_true",
                    help="Use the preprocessed & saved features & outputs. "
                         "Need to first run with '--save'. "
                         "Options `--woh`, `--doh`, and `--no-loc-id` have "
                         "to be specified the same way as when the data has "
                         "been saved")
parser.add_argument("--save", action="store_true",
                    help="Save the np/sparse array containing cross-superboro "
                         "trips to disk")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Let the model print training progress (if supported)")


def crossboro_preproc(features, doh, woh, loc_id):
    """Given the details of a single cross-superboro trip
    as a 1-D array of features, for each bridge connecting
    the two superboros involved, return the following info:
        - PU Super-boro code (int between 1 and 3)
        - DO Super-boro code (int between 1 and 3)
        - Features for PU->bridge (np.array)
        - Features for bridge->DO (np.array)

    :features: A single row in `features` array obtained
        from `extract_features` function call
        (corresponds to a single cross-superboro trip)
    :doh: Boolean for datetime-one-hotness
    :woh: Boolean for weekdays-one-hotness
    :loc_id: Boolean for including PU, DO locationIDs
        (locationIDs are one-hot if included)
    :returns: A list with length equal to the number of
        bridges between the two super-boros, where each
        element is a sublist with the info listed above
    """
    # NOTE: if `features` is a row from `scipy.sparse` matrix,
    #       it may have to be converted into `np.array` (dense)        
    is_sparse = doh or woh or loc_id

    raise NotImplementedError("ERROR: Cross-boro preprocessing "
                               "has not yet been implemented")

def load_models(args):
    """Returns a list containing XGBoost models to
    conduct cross-super-boro inferences with.
    The `None` at the beginning is added as a padding,
    to match model indices with superboro codes.

    :args: argparse Namespace
    :returns: List containing a `None` at index 0,
        followed by three XGBoost models
    """
    raise NotImplementedError("ERROR: Choose models to load for inference")
    sb1_model = xgb.load_model(args.sb1_model_path)
    sb2_model = xgb.load_model(args.sb2_model_path)
    sb3_model = xgb.load_model(args.sb3_model_path)
    return [None,sb1_model,sb2_model,sb3_model]


def evaluate(models, features, outputs, doh, woh, loc_id):
    """Evaluate the selected superboro models on cross-superboro
    trips.

    :models: List of models for prediction (starting at index 1)
    :features, outputs: 
    """
    total_loss = 0
    convert = lambda trip: crossboro_preproc(trip, doh, woh, loc_id)

    # Iterate through each cross-superboro trip
    for inputs, output in zip(features, outputs):
        inputs = [convert(trip) for trip in inputs]

        min_loss = 1e20

        # Compute loss for each bridge and record minimum
        for (sb_PU, sb_DO, f_PU, f_DO) in inputs:
            f_PU_dmat = xgb.DMatrix(f_PU)
            f_DO_dmat = xgb.DMatrix(f_DO)

            # Compute durations for each trip
            PU_duration = models[sb_PU].predict(f_PU_dmat)
            DO_duration = models[sb_DO].predict(f_DO_dmat)

            # Compute total duration & MSE loss
            pred = PU_duration + DO_duration
            loss = (pred - outputs)**2
            min_loss = min(min_loss, loss)

        total_loss += min_loss

    return np.sqrt(total_loss)


def load_cross_superboro(args):
    """Load cross-superboro datapoints from DB into memory,
    with features as `scipy.sparse.csr_matrix` or `np.array`
    and outputs as `np.array`.
    Iterates through each pair of distinct combinations of
    super-boroughs and stacks the corresponding features and
    outputs together.
    Saves the stacked features and outputs to disk if `--save`
    option has been specified.

    :args: argparse Namespace
    :returns: features and outputs arrays containing all
        cross-superboro trips
    """
    conn = create_connection(args.db_path)

    is_sparse = args.datetime_one_hot \
                or args.weekdays_one_hot \
                or args.loc_id
    features = None
    outputs = None

    if args.verbose:
        start_time = time()

    for start_sb, end_sb in ([1,2],[1,3],[2,3]):
        if args.verbose:
            print(f">>> Working on SBs {start_sb} and {end_sb}...")

        data_params = {
            "table_name":"rides",
            "variant":"random" if args.rand_subset > 0 else "all",
            "size":args.rand_subset,
            "datetime_onehot":args.datetime_one_hot,
            "weekdays_onehot":args.weekdays_one_hot,
            "include_loc_ids":args.loc_id,
            "start_super_boro":SUPERBORO_CODE[start_sb],
            "end_super_boro":SUPERBORO_CODE[end_sb],
            "stddev_multiplier":args.stddev_mul,
        }

        extracted_features, extracted_outputs = \
            extract_features(conn, **data_params)

        if args.verbose:
            print(">>> Extraction complete, "
                  f"# of rows: {extracted_outputs.shape[0]}")

        if features is None and outputs is None:
            features = extracted_features
            outputs = extracted_outputs
        else:
            if is_sparse:
                features = sparse.vstack([features, extracted_features],
                                         format="csr")
            else:
                features = np.vstack([features, extracted_features])
            outputs = np.concatenate([outputs, extracted_outputs])

    if args.verbose:
        print(">>> All pairs complete, "
              f"# of rows: {outputs.shape[0]}, "
              f"total duration: {time() - start_time:.2f} seconds")

    if args.save:
        f_path = "./data/crossboro_" \
                 f"{int(args.datetime_one_hot)}" \
                 f"{int(args.weekdays_one_hot)}" \
                 f"{int(args.loc_id)}" \
                 "_features"
        o_path = "./data/crossboro_" \
                 f"{int(args.datetime_one_hot)}" \
                 f"{int(args.weekdays_one_hot)}" \
                 f"{int(args.loc_id)}" \
                 "_outputs.npy"

        if is_sparse:
            sparse.save_npz(f_path, features)
        else:
            np.save(f_path, features)
        np.save(o_path, outputs)

        if args.verbose:
            print(">>> Features and outputs saved to disk")

    return features, outputs


def main():
    args = parser.parse_args()
    is_sparse = args.datetime_one_hot \
                or args.weekdays_one_hot \
                or args.loc_id

    if args.use_saved:  # Load arrays stored in disk
        f_path = "./data/crossboro_" \
                 f"{int(args.datetime_one_hot)}" \
                 f"{int(args.weekdays_one_hot)}" \
                 f"{int(args.loc_id)}" \
                 "_features"
        o_path = "./data/crossboro_" \
                 f"{int(args.datetime_one_hot)}" \
                 f"{int(args.weekdays_one_hot)}" \
                 f"{int(args.loc_id)}" \
                 "_outputs.npy"
        if args.verbose:
            start_time = time()

        features = sparse.load_npz(f_path + ".npz") if is_sparse \
                    else np.load(f_path + ".npy")
        outputs = np.load(o_path)

        if args.verbose:
            print(">>> Loading from disk complete, "
                  f"# of rows: {outputs.shape[0]}, "
                  f"total duration: {time() - start_time:.2f} seconds")
            
        args = parse_dmat_name(args)
    else:   # Parse arrays from DB
        features, outputs = load_cross_superboro(args)

    if args.verbose:
        print(f">>> features.shape = {features.shape}")
        print(f">>> outputs.shape = {outputs.shape}")

    return # TODO: Delete once the below code is ready to run

    models = load_models(args)
    loss = evaluate(models, features, outputs,
                    args.datetime_one_hot,
                    args.weekdays_one_hot,
                    args.loc_id)
    print(f">>> Loss for cross-superboro trips: {loss}")


if __name__ == "__main__":
    main()
