""" Contains code to read CSV files containing taxi data """
from math import ceil
from glob import glob
import sqlite3
import argparse
import pandas as pd
from tqdm import tqdm
import geojson
import numpy as np

DATE_COLUMNS = [
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime'
    ]

PARSER = argparse.ArgumentParser(description='Parse TaxiData CSVs to SQL')
PARSER.add_argument('--rebuild_rides_table',
                    help='Whether or not to rebuild the rides table. '
                         'Requires argument: string containing regex matching '
                         'rides CSVs to parse'
                   )

PARSER.add_argument('--rebuild_locations_table',
                    help='Whether or not to rebuild the locations table. '
                         'Requires argument: string containing path to '
                         'locations CSV'
                   )

PARSER.add_argument('--rebuild_coordinates_table',
                    help='Whether or not to rebuild the coordinates table. '
                         'Requires argument: string containing path to '
                         'geojson file containing zone info'
                   )



def parse_files(file_list, convert_date_time=True):
    """Parses the context of files in file_list

    :file_regex: a list of files whose contents should be parsed
    :returns: A pandas dataframe containing every row in the specified files
    """

    all_data_frames = (pd.read_csv(f, header=0) for f in file_list)
    merged_data = pd.concat((all_data_frames))

    if convert_date_time:
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


def parse_geo_json(file_path):
    """Parses a geo_json file into our database

    :file_path: The file containing the geojson which we wish to parse
    :returns: A pandas dataframe containing the zone ID and mean lat/long
        for each zone.

    """
    with open(file_path[0]) as opened_file:
        loaded_geojson = geojson.loads(opened_file.read())
        print(loaded_geojson['type'])
        for zone in loaded_geojson['features']:
            location_id = zone['properties']['location_id']
            coordinates = np.array(list(geojson.utils.coords(zone)))

            # Parsing sometimes adds useless dimensions for some reason.
            # Remove them.
            maxes = np.max(coordinates, axis=0)
            mins = np.min(coordinates, axis=0)
            mean_lat, mean_long = maxes + mins / 2


def parse_files_and_write_to_db(file_regex,
                                db_conn,
                                table_name,
                                chunk_size=5,
                                convert_date_time=True,
                                geojson=False):
    """
    Parses the given files and writes them to a table in a database.

    :file_regex: a regular expression matching the files whose contents we
        want to parse
    :db_conn: The database connection object
    :table_name: The name of the table to which we will write
    :chunk_size: The number of files to write to the db at once
    :convert_date_time: Whether or not to attempt to convert certain columns
        to datetimes
    :geojson: Whether or not to parse file_regex as geojson files
    """
    empty_table(db_conn, table_name)

    matching_files = glob(file_regex)
    num_chunks = ceil(len(matching_files)/chunk_size)
    with tqdm(total=num_chunks) as pbar:
        for chunk in chunk_iter(matching_files, chunk_size):
            if not geojson:
                all_data = parse_files(chunk, convert_date_time)
            else:
                all_data = parse_geo_json(chunk)
            write_to_db(all_data, db_conn, table_name)
            pbar.update(1)


def main():
    """Handles args and calls parse_files_and_write_to_db.

    """
    db_conn = sqlite3.connect('rides.db')

    provided_args = PARSER.parse_args()

    if provided_args.rebuild_rides_table:
        table_name = 'rides'
        parse_files_and_write_to_db(provided_args.rebuild_rides_table,
                                    db_conn,
                                    table_name)

    if provided_args.rebuild_locations_table:
        table_name = 'locations'
        parse_files_and_write_to_db(provided_args.rebuild_locations_table,
                                    db_conn,
                                    table_name,
                                    convert_date_time=False)

    if provided_args.rebuild_coordinates_table:
        table_name = 'coordinates'
        parse_files_and_write_to_db(provided_args.rebuild_coordinates_table,
                                    db_conn,
                                    table_name,
                                    convert_date_time=False,
                                    geojson=True)



if __name__ == '__main__':
    main()
