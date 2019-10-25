""" Contains code to read CSV files containing taxi data """
import sys
from glob import glob
import csv

def parse_files(file_regex, total_rows=None):
    """Parses the context of files matching file_regex

    :file_regex: a regular expression matching the files whose contents we
        want to parse
    :total_rows: if not None, the total number of rows to return. Will read this
        many and return early.
    :returns: A list of lists, each inner list containing a csv row

    """
    matching_files = glob(file_regex)

    all_rows = []

    rows_read = 0

    for current_file in matching_files:
        with open(current_file) as opened_file:
            file_reader = csv.reader(opened_file)

            # Skip the header row
            next(file_reader)

            for row in file_reader:
                all_rows.append(row)
                rows_read += 1

                if total_rows and rows_read == total_rows:
                    break

            if total_rows and rows_read == total_rows:
                break

    print(len(all_rows))
    return all_rows

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Please supply a path to files to parse')
    parse_files(sys.argv[1], total_rows=1234)
