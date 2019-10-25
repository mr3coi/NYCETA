""" Contains code to read CSV files containing taxi data """
import sys
from glob import glob
import csv

def parse_files(file_regex):
    """Parses the context of files matching file_regex

    :file_regex: a regular expression matching the files whose contents we
        want to parse
    :returns: A list of lists, each inner list containing a csv row

    """
    matching_files = glob(file_regex)

    all_rows = []

    for current_file in matching_files:
        with open(current_file) as opened_file:
            file_reader = csv.reader(opened_file)

            # Skip the header row
            next(file_reader)

            for row in file_reader:
                all_rows.append(row)

    return all_rows

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Please supply a path to files to parse')
    parse_files(sys.argv[1])
