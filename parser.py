""" Contains code to read CSV files containing taxi data """
import sys
from math import ceil
from glob import glob
import sqlite3
import pandas as pd
from tqdm import tqdm

DATE_COLUMNS = [
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime'
    ]

def parse_files(file_list):
    """Parses the context of files in file_list

    :file_regex: a list of files whose contents should be parsed
    :returns: A pandas dataframe containing every row in the specified files
    """

    all_data_frames = (pd.read_csv(f, header=0) for f in file_list)
    merged_data = pd.concat((all_data_frames))

    for col in DATE_COLUMNS:
        merged_data[col] = pd.to_datetime(merged_data[col])

    return merged_data


def chunk_iter(source, chunk_size):
    """Iterates over an iterable, dividing it into chunks of size chunk_size.
    If the length of source is not a multiple of chunk_size, then the last
    chunk yielded will not be of size chunk_size, and will instead contain
    all elements that have not yet been yielded.

    :source: the source iterable
    :chunk_size: the size of the lists to yield
    :yields: iterables of size chunk_size consisting of elements from source
    """
    for index in range(0, len(source), chunk_size):
        yield source[index:index+chunk_size]


def empty_table(conn, table_name):
    """Deletes all rows in a table specified by table_name in a database
    associated with connection conn.

    :conn: a sqlite3 database connection
    :table_name: the name of the table to delete
    """
    deletion_command = f'DROP TABLE IF EXISTS {table_name};'
    cur = conn.cursor()
    cur.execute(deletion_command)
    conn.commit()


def write_to_db(dataframe, db_conn, table_name):
    """Writes the given dataframe to a database table. If the table
    already exists, it will overwrite its contents.

    :dataframe: The dataframe to write to a database table
    :db_conn: The database connection object
    :table_name: The name of the table to which we will write
    """
    dataframe.to_sql(table_name, db_conn, if_exists='append')


def parse_files_and_write_to_db(file_regex, db_conn, table_name, chunk_size=5):
    """
    Parses the given files and writes them to a table in a database.

    :file_regex: a regular expression matching the files whose contents we
        want to parse
    :db_conn: The database connection object
    :table_name: The name of the table to which we will write
    :chunk_size: The number of files to write to the db at once
    """
    empty_table(db_conn, table_name)

    matching_files = glob(file_regex)
    num_chunks = ceil(len(matching_files)/chunk_size)
    with tqdm(total=num_chunks) as pbar:
        for chunk in chunk_iter(matching_files, chunk_size):
            all_data = parse_files(chunk)
            write_to_db(all_data, db_conn, table_name)
            pbar.update(1)


def main(file_regex):
    """Calls parse_files_and_write_to_db.

    :file_regex: The regex containing the files to parse.
    """
    table_name = 'rides'
    db_conn = sqlite3.connect('rides.db')
    parse_files_and_write_to_db(file_regex, db_conn, table_name)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Please supply a path to files to parse')
    main(sys.argv[1])
