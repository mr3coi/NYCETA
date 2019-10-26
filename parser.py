""" Contains code to read CSV files containing taxi data """
import sys
from glob import glob
import pandas as pd


def parse_files(file_regex):
    """Parses the context of files matching file_regex

    :file_regex: a regular expression matching the files whose contents we
        want to parse
    :returns: A pandas dataframe containing every row in the specified files

    """
    matching_files = glob(file_regex)

    all_data_frames = (pd.read_csv(f, header=0) for f in matching_files)
    merged_data = pd.concat((all_data_frames))

    return merged_data

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Please supply a path to files to parse')
    parse_files(sys.argv[1], total_rows=1234)
