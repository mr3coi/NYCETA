""" Contains code to read CSV files containing taxi data """
import sys
from glob import glob
import pandas as pd
import sqlite3

DATE_COLUMNS = [
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime'
    ]

def parse_files(file_regex):
    """Parses the context of files matching file_regex

    :file_regex: a regular expression matching the files whose contents we
        want to parse
    :returns: A pandas dataframe containing every row in the specified files

    """
    matching_files = glob(file_regex)

    all_data_frames = (pd.read_csv(f, header=0) for f in matching_files)
    merged_data = pd.concat((all_data_frames))

    for col in DATE_COLUMNS:
        merged_data[col] = pd.to_datetime(merged_data[col])

    return merged_data


def write_to_db(dataframe, db_conn, table_name):
    """Writes the given dataframe to a database table. If the table
    already exists, it will overwrite its contents.

    :dataframe: The dataframe to write to a database table
    :db_conn: The database connection object
    :table_name: The name of the table to which we will write

    """
    dataframe.to_sql(table_name, db_conn, if_exists='replace')


def parse_files_and_write_to_db(file_regex, db_conn, table_name):
    """
    Parses the given files and writes them to a table in a database.

    :file_regex: a regular expression matching the files whose contents we
        want to parse
    :db_conn: The database connection object
    :table_name: The name of the table to which we will write

    """
    all_data = parse_files(file_regex)
    write_to_db(all_data, db_conn, table_name)


def main(file_regex):
    """
    Calls parse_files_and_write_to_db.

    :file_regex: The regex containing the files to parse.

    """
    table_name = 'rides'
    db_conn = sqlite3.connect('rides.db')
    parse_files_and_write_to_db(file_regex, db_conn, table_name)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Please supply a path to files to parse')
    main(sys.argv[1])
