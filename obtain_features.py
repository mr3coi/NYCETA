# pragma pylint: disable=C0103, C0303
import sqlite3
import sys
import datetime
import numpy as np
from scipy import sparse
from utils import create_connection


def get_one_hot(values, min_val, max_val):
    """Obtain a one-hot encoding for the given value

    :values: list or np array of the values to be encoded
    :min_val: the minimum value possible
    :max_val: the maximum value possible
    :return: a numpy array of the one hot encoding
    """
    rows = np.arange(len(values))
    cols = np.array([i-min_val for i in values])
    data = np.ones(len(values))

    enc = sparse.csr_matrix((data, (rows, cols)), shape=(len(values), max_val-min_val+1))

    return enc


def parse_datetime(datetime_strs):
    """Obtain the numerical values of dates, months,
    hours, minutes and seconds from the datetime strings

    :datetime_strs: a list of datetime_strs of parse
    :returns: a list of lists of dates, months, hours,
        minutes and seconds
    """
    dt_list = [i.split(" ") for i in datetime_strs]
    dates, times = map(list, zip(*dt_list))
   
    date_list = [i.split("-") for i in dates]
    
    dates = [int(i[2]) for i in date_list]
    months = [int(i[1]) for i in date_list]
    # The year information will only be used to get the day of the week
    years = [int(i[0]) for i in date_list]
    weekdays = [datetime.date(years[i], months[i], dates[i]).weekday \
        for i in range(len(date_list))]

    time_list = [i.split(":") for i in times]
    hours = [int(i[0]) for i in time_list]
    minutes = [int(i[1]) for i in time_list]
    seconds = [int(i[2]) for i in time_list]

    return [dates, months, hours, minutes, seconds, weekdays]


def obtain_date_time_features(datetime_lists):
    """Parses a string the standard datetime format
    yyyy-mm-dd hh:mn:ss
    to obtain a feature vector from it

    :datetime_lists:  a list of lists of dates, months, hours,
        minutes and seconds
    :returns: a numpy array of dim (list_size, feature_length)
    """
    dates = get_one_hot(datetime_lists[0], 1, 31)
    months = get_one_hot(datetime_lists[1], 1, 12)

    hours = get_one_hot(datetime_lists[2], 0, 23)
    minutes = get_one_hot(datetime_lists[3], 0, 59)
    seconds = get_one_hot(datetime_lists[4], 0, 59)
    
    weekdays = get_one_hot(datetime_lists[5], 0, 6)

    features = sparse.hstack([dates, months, hours, minutes, seconds, weekdays])
    return features


def get_naive_features(rows, maxLocID=265):
    """Obtain the naive features to which contain
    the all the information available to us
    in the concatanted vector

    :rows: the rows of data obtained from the database
    :maxLocID: the maximum possibel value of location IDs
    :returns: a sparse csr_matrix for the feature vectors, 
        and a np array for the time taken in seconds
    """
    p_datetime = parse_datetime(rows[:, 0])
    d_datetime = parse_datetime(rows[:, 1])
    
    PUDatetime = obtain_date_time_features(p_datetime)
    PULocID = get_one_hot(list(map(int, rows[:, 2])), 1, maxLocID)
    DOLocID = get_one_hot(list(map(int, rows[:, 3])), 1, maxLocID)
    feature_vectors = sparse.hstack([PUDatetime, PULocID, DOLocID])
    
    delta = np.array([datetime.timedelta(
        days=d_datetime[0][i]-p_datetime[0][i],
        hours=d_datetime[2][i]-p_datetime[2][i],
        minutes=d_datetime[3][i]-p_datetime[3][i],
        seconds=d_datetime[4][i]-p_datetime[4][i]
    ) for i in range(rows.shape[0])])

    time_taken = [i.seconds for i in delta]
    
    return feature_vectors, time_taken


def extract_all_features(conn, table_name):
    """Extracts the features from all the data entries 
    in the given table of the database

    :conn: connection object to the database
    :table_name: name of the table holding the data
    :returns: a sparse csr_matrix containing the feature vectors
        and a numpy array containing the corresponding values
        of the travel time
    """
    offset = 0
    limit = 1e6
    batch_num = 0
    cursor = conn.cursor()

    print("Extracting data in batches of size {}".format(limit))
    stop_condition = False
    while True:
        print("Reading data for batch number {}".format(batch_num))
        command = ('SELECT tpep_pickup_datetime, tpep_dropoff_datetime, '
                   'PULocationID, DOLocationID '
                   f'FROM {table_name} '
                   f'LIMIT {limit} '
                   f'OFFSET {offset}')
        
        try:
            cursor.execute(command)
        except sqlite3.Error as e:
            print(e)
            stop_condition = True
        if stop_condition:
            break
        
        rows = np.array(cursor.fetchall())  

        print("Extracting features from the read data")
        if offset == 0:
            features, outputs = get_naive_features(rows)
        else:
            features_sample, outputs_sample = get_naive_features(rows) 
            features = sparse.vstack([features, features_sample])
            outputs = np.concatenate((outputs, outputs_sample))

        batch_num += 1
        offset += limit
    return features, outputs


def extract_random_data_features(conn, table_name, random_size):
    """Extracts the features from a random batch of data 
    from the table of the database

    :conn: connection object to the database
    :table_name: name of the table holding the data
    :random_size: the size of the random batch to be taken
    :returns: a sparse csr_matrix containing the feature vectors
        and a numpy array containing the corresponding values
        of the travel time
    """
    cursor = conn.cursor()

    command = ('SELECT tpep_pickup_datetime, tpep_dropoff_datetime, '
               'PULocationID, DOLocationID '
               f'FROM {table_name} '
               'ORDER BY RANDOM() '
               f'LIMIT {random_size}')
    print('Reading data entries from the table in the database')
    try:
        cursor.execute(command)
    except Error as e:
        print(e)

    rows = np.array(cursor.fetchall())

    print("Making feature vectors from the extracted data")
    features, outputs = get_naive_features(rows)

    return features, outputs


def extract_features(conn, table_name, variant='all', random_size=None):
    """Reads the data from the database and obtains the features

    :conn: connection object to the database
    :table_name: name of the table holding the data
    :variant: which type of variant to choose for extracting data
        Must be one out of
            - all : extracts features from all the data
            - random : uses a random batch of data from the db
    :random_size: the size of the random batch of data 
        (Used only if variant='random')
    :returns: a sparse csr_matrix containing the feature vectors
        and a numpy array containing the corresponding values
        of the travel time
    """
    if variant == 'all':
        print('Extracting features from all the data in {}'.format(table_name))
        features, outputs = extract_all_features(conn, table_name)

    elif variant == 'random':
        if not type(random_size) is int:
            print('Please provide an integer size for the random batch.')
        print('Extracting features from a random batch of data of size {} in {}'.format(random_size, table_name))
        features, outputs = extract_random_data_features(conn, table_name, random_size)
    
    else:
        print("Type must be one of {'all', 'random'}")
        sys.exit(0)

    return features, outputs


if __name__ == "__main__":
    db_name = "rides.db" 
    con = create_connection(db_name)   
    # We have a total of 67302302 entries in the rides table 
    features_, outputs_ = extract_features(con, "rides", variant='all', random_size=10)
    print(features_.shape)
    print(outputs_.shape)
