""" Contains code to read CSV files containing taxi data """
import sys
from glob import glob

def parse_files(file_regex):
    """Parses the context of files matching file_regex

    :file_regex: TODO
    :returns: TODO

    """
    print(glob(file_regex))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Please supply a path to files to parse')
    parse_files(sys.argv[1])
