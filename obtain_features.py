# pragma pylint: disable=C0103, C0303
import sqlite3
from sqlite3 import Error
import sys
import datetime
import numpy as np
import borough_labels
from scipy import sparse
from utils import create_connection
from math import floor
from time import time
from sklearn.preprocessing import MinMaxScaler


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
    # weekdays range from 1-7
    weekdays = [datetime.date(years[i], months[i], dates[i]).weekday()+1 \
        for i in range(len(date_list))]

    time_list = [i.split(":") for i in times]
    hours = [int(i[0]) for i in time_list]
    minutes = [int(i[1]) for i in time_list]
    seconds = [int(i[2]) for i in time_list]

    return [dates, months, hours, minutes, seconds, weekdays]


def obtain_date_time_features(datetime_lists, datetime_onehot=True, weekdays_onehot=True):
    """Parses a string the standard datetime format
    yyyy-mm-dd hh:mn:ss
    to obtain a feature vector from it

    :datetime_lists:  a list of lists of dates, months, hours,
        minutes and seconds
    :datetime_onehot: boolean for whether we want a onehot represnetation for
        date and time values, or a single index one
    :weekdays_onehot: boolean for whether we want a onehot represnetation for
        day of the week value, or a single index one
    :returns: a numpy array of dim (list_size, feature_length) or a sparse matrix 
        if we use any one hot representation
    """
    if datetime_onehot:
        dates = get_one_hot(datetime_lists[0], 1, 31)
        months = get_one_hot(datetime_lists[1], 1, 12)

        hours = get_one_hot(datetime_lists[2], 0, 23)
        minutes = get_one_hot(datetime_lists[3], 0, 59)
        seconds = get_one_hot(datetime_lists[4], 0, 59)
        
    else:
        dates = np.array(datetime_lists[0])
        dates = np.reshape(dates, (dates.shape[0], 1))
        months = np.array(datetime_lists[1])
        months = np.reshape(months, (months.shape[0], 1))

        # if the representation isnt one hot, we make the range 
        # for hours, minutes and seconds, start from 1 instead of 0. 
        hours = np.array(datetime_lists[2])+1
        hours = np.reshape(hours, (hours.shape[0], 1))
        minutes = np.array(datetime_lists[3])+1
        minutes = np.reshape(minutes, (minutes.shape[0], 1))
        seconds = np.array(datetime_lists[4])+1
        seconds = np.reshape(seconds, (seconds.shape[0], 1))

    if weekdays_onehot:
        weekdays = get_one_hot(datetime_lists[5], 1, 7)
    else:
        weekdays = np.array(datetime_lists[5])
        weekdays = np.reshape(weekdays, (weekdays.shape[0], 1))

    if datetime_onehot or weekdays_onehot: 
        features = sparse.hstack([dates, months, hours, minutes, seconds, weekdays], format="csr")
    else:
        features = np.hstack([dates, months, hours, minutes, seconds, weekdays])
        # features = sparse.csr_matrix(features)

    return features


def extract_all_coordinates(conn, table_name, normalized=True):
    """Extracts the mean coordinates of all the zones
    from the coordinates table

    :conn: connection object to the database 
    :table_name: name of the table holding the coordinates data
    :normalized: boolean for whether to normalize each coordinate 
        between 0 and 1 or not (i.e. retain actual values)
    :returns: a 2D numpy array where the entry at index i is 
        the coordinates of location i
    """
    cursor = conn.cursor()
    command = f'SELECT LocationID, lat, long FROM {table_name}'
    cursor.execute(command)
    rows = np.array(cursor.fetchall(), dtype='int, float32, float32')
    # obtain the coordinates in a (num_zones x 2)-dim np array
    # where the the 2 dim array at each index i
    # is the coordinates of location ID i.
    coords = np.zeros((rows.shape[0], 2))    
    for row in rows:
        coords[row[0]-1] = [row[1], row[2]]

    # Normalizing entries between maximum and minimum 
    if normalized:
        mm_scaler = MinMaxScaler()
        coords = mm_scaler.fit_transform(coords)
        print('Coordinate scaling minimums are {}'.format(mm_scaler.data_min_))
        print('Coordinate scaling ranges are {}'.format(mm_scaler.data_range_))

    # to make the indexes start from 1, we add dummy entry at index 0 
    coords = np.vstack([np.zeros(2), coords])

    return coords


def extract_all_boroughs(conn, table_name, maxLocID=263):
    """Extracts the boroughs of all the zones
    from the coordinates table

    :conn: connection object to the database 
    :table_name: name of the table holding the boroughs data
    :maxLocID: the maximum possible value of location IDs
    :returns: a 2D numpy array where the entry at index i is 
        the one-hot label for borough of location i
    """

    cursor = conn.cursor()
    command = f'SELECT LocationID, Borough FROM {table_name}'
    cursor.execute(command)
    rows = np.array(cursor.fetchall())

    boroughs = np.zeros(maxLocID)
    for row in rows:
        if row[1] in borough_labels.BOROUGHS:
            boroughs[int(row[0])-1] = borough_labels.BOROUGHS[row[1]]

    # get the one-hot representation for the borough labels
    boroughs = get_one_hot(boroughs, 1, 6).toarray()

    # to make the indexes start from 1, we add dummy entry at index 0 
    boroughs = np.vstack([np.zeros(6), boroughs])

    return boroughs


def get_naive_features(rows, coords, boros, maxLocID=263, datetime_onehot=True, 
    weekdays_onehot=True, include_loc_ids=True):
    """Obtain the naive features to which contain
    the all the information available to us
    in the concatanted vector

    :rows: the rows of data obtained from the database
    :coords: list of coordinates of locations, where index i
        holds the coordinates of locationID i
    :boros: list of borough labels of locations, where index i
        holds the one-hot label for borough of locationID i
    :maxLocID: the maximum possible value of location IDs
    :datetime_onehot: boolean for whether we want a onehot represnetation for
        date and time values, or a single index one
    :weekdays_onehot: boolean for whether we want a onehot represnetation for
        day of the week value, or a single index one
    :include_loc_ids: boolean for whether to include locIds as one-hot
        in the feature vectors, or not 
    :returns: a sparse csr_matrix for the feature vectors(if we use any hot rep)
        or a numpy array for the features vectors, 
        and a np array for the time taken in seconds
    """

    p_datetime = parse_datetime(rows[:, 0])
    d_datetime = parse_datetime(rows[:, 1])
    
    PUDatetime = obtain_date_time_features(p_datetime, datetime_onehot, weekdays_onehot)
    PULocID = list(map(int, rows[:, 2]))
    DOLocID = list(map(int, rows[:, 3]))
    PUCoords = np.array([coords[i] for i in PULocID])
    DOCoords = np.array([coords[i] for i in DOLocID])
    PUBoroughs = np.array([boros[i] for i in PULocID])
    DOBoroughs = np.array([boros[i] for i in DOLocID])
    
    if include_loc_ids:
        PULocID = get_one_hot(PULocID, 1, maxLocID)
        DOLocID = get_one_hot(DOLocID, 1, maxLocID)
        feature_vectors = sparse.hstack([PUDatetime, PUCoords, DOCoords, 
            PUBoroughs, DOBoroughs, PULocID, DOLocID], format="csr")
    elif datetime_onehot or weekdays_onehot:
        feature_vectors = sparse.hstack([PUDatetime, PUCoords, DOCoords, PUBoroughs, DOBoroughs], format="csr")
    else:
        feature_vectors = np.hstack([PUDatetime, PUCoords, DOCoords, PUBoroughs, DOBoroughs])

    delta = np.array([datetime.timedelta(
        days=d_datetime[0][i]-p_datetime[0][i],
        hours=d_datetime[2][i]-p_datetime[2][i],
        minutes=d_datetime[3][i]-p_datetime[3][i],
        seconds=d_datetime[4][i]-p_datetime[4][i]
    ) for i in range(rows.shape[0])])

    time_taken = np.array([i.seconds for i in delta])
    
    return feature_vectors, time_taken


def extract_all_features(conn, table_name, coords_table_name='coordinates', boros_table_name='locations',
    datetime_onehot=True, weekdays_onehot=True, include_loc_ids=True):
    """Extracts the features from all the data entries 
    in the given table of the database

    :conn: connection object to the database
    :table_name: name of the table holding the rides data
    :coords_table_name: name of the table holding the coordinates data
    :boross_table_name: name of the table holding the boroughs data
    :datetime_onehot: boolean for whether we want a onehot represnetation for
        date and time values, or a single index one
    :weekdays_onehot: boolean for whether we want a onehot represnetation for
        day of the week value, or a single index one
    :include_loc_ids: boolean for whether to include locIds as one-hot
        in the feature vectors, or not 
    :returns: a sparse csr_matrix containing the feature vectors
        and a numpy array containing the corresponding values
        of the travel time
    """
    offset = 0
    limit = 1e6
    batch_num = 0
    cursor = conn.cursor()

    # extracting coordinates of all 
    coords = extract_all_coordinates(conn, coords_table_name)

    # extracting boroughs for all locations
    boros = extract_all_boroughs(conn, boros_table_name)

    print("Extracting data in batches of size {}".format(limit))
    stop_condition = False
    while True:
        print("Reading data for batch number {}".format(batch_num))
        command = ('SELECT tpep_pickup_datetime, tpep_dropoff_datetime, '
                   'PULocationID, DOLocationID '
                   f'FROM {table_name} '
                   f'WHERE PULocationID < 264 '
                   f'AND DOLocationID < 264 '
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
        if rows.shape[0] == 0:
            break

        print("Extracting features from the read data")
        if offset == 0:
            features, outputs = get_naive_features(rows, coords, boros, 
                                    datetime_onehot=datetime_onehot, 
                                    weekdays_onehot=weekdays_onehot, 
                                    include_loc_ids=include_loc_ids)
        else:
            features_sample, outputs_sample = get_naive_features(rows, coords, boros, 
                                                datetime_onehot=datetime_onehot, 
                                                weekdays_onehot=weekdays_onehot, 
                                                include_loc_ids=include_loc_ids)
            features = sparse.vstack([features, features_sample], format="csr")
            outputs = np.concatenate((outputs, outputs_sample))

        batch_num += 1
        offset += limit
    return features, outputs


def extract_random_data_features(conn, table_name, random_size, 
    coords_table_name='coordinates', boros_table_name='locations', 
    datetime_onehot=True, weekdays_onehot=True, include_loc_ids=True):
    """Extracts the features from a random batch of data 
    from the table of the database

    :conn: connection object to the database
    :table_name: name of the table holding the rides data
    :random_size: the size of the random batch to be taken
    :coords_table_name: name of the table holding the coordinates data
    :boross_table_name: name of the table holding the boroughs data
    :datetime_onehot: boolean for whether we want a onehot represnetation for
        date and time values, or a single index one
    :weekdays_onehot: boolean for whether we want a onehot represnetation for
        day of the week value, or a single index one
    :include_loc_ids: boolean for whether to include locIds as one-hot
        in the feature vectors, or not 
    :returns: a sparse csr_matrix containing the feature vectors
        and a numpy array containing the corresponding values
        of the travel time
    """
    cursor = conn.cursor()

    # extracting coordinates of all 
    coords = extract_all_coordinates(conn, coords_table_name)

    # extracting boroughs for all locations
    boros = extract_all_boroughs(conn, boros_table_name)

    command = ('SELECT tpep_pickup_datetime, tpep_dropoff_datetime, '
               'PULocationID, DOLocationID '
               f'FROM {table_name} '
               f'WHERE PULocationID < 264 '
               f'AND DOLocationID < 264 '
               'ORDER BY RANDOM() '
               f'LIMIT {random_size}')
    print('Reading data entries from the table in the database')
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print(e)

    rows = np.array(cursor.fetchall())

    print("Making feature vectors from the extracted data")
    features, outputs = get_naive_features(rows, coords, boros, datetime_onehot=datetime_onehot, 
                            weekdays_onehot=weekdays_onehot, include_loc_ids=include_loc_ids)

    return features, outputs


def extract_batch_features(conn, table_name, batch_size, block_size,
    coords_table_name='coordinates', boros_table_name='locations',
    datetime_onehot=True, weekdays_onehot=True, include_loc_ids=True,
    replace_blk=False, verbose=False):
    """Extracts the features from a batch of data
    from the table of the database, without shuffling

    :conn: connection object to the database
    :table_name: name of the table holding the rides data
    :batch_size: the size of the batch to be taken
    :block_size: the size of each block(chunk) of rows that constitute
        a single batch. Determines the granularity of shuffling.
    :coords_table_name: name of the table holding the coordinates data
    :boross_table_name: name of the table holding the boroughs data
    :datetime_onehot: boolean for whether we want a onehot represnetation for
        date and time values, or a single index one
    :weekdays_onehot: boolean for whether we want a onehot represnetation for
        day of the week value, or a single index one
    :include_loc_ids: boolean for whether to include locIds as one-hot
        in the feature vectors, or not 
    :replace_blk: whether to sample blocks with/without replacement
        when forming a minibatch
    :verbose: whether to print out progress onto stdout
    :returns: a generator that yields each minibatch
        as a (features, outputs) pair.
    """
    cursor = conn.cursor()

    # extracting coordinates of all locations
    coords = extract_all_coordinates(conn, coords_table_name)

    # extracting boroughs for all locations
    boros = extract_all_boroughs(conn, boros_table_name)

    if verbose:
        count_start = time()
    count_cmd = (f'SELECT COUNT(PULocationID) FROM {table_name} ')
    try:
        cursor.execute(count_cmd)
    except Error as e:
        print(e)
    NUM_ROWS = cursor.fetchone()[0]
    assert type(NUM_ROWS) is int, f'cursor.fetchone() has returned {type(NUM_ROWS)} instead of an int.'

    NUM_BATCH = floor(NUM_ROWS / batch_size)

    NUM_BLKS = floor(NUM_ROWS / block_size)
    BLKS_PER_BATCH = (int)(batch_size / block_size)
    blk_list = np.arange(NUM_BLKS)

    for batch_idx in range(NUM_BATCH):
        print(f"Loading batch {batch_idx+1}/{NUM_BATCH}")
        blks_in_sample = np.random.choice(NUM_BLKS,
                                          size=BLKS_PER_BATCH,
                                          replace=False)
        if not replace_blk:
            blk_list = blk_list[[item not in blks_in_sample for item in blk_list]]

        for i, blk_idx in enumerate(np.sort(blks_in_sample)):
            if verbose:
                print(f">>> Loading block {i+1}/{BLKS_PER_BATCH} of the minibatch")

            command = ('SELECT tpep_pickup_datetime, tpep_dropoff_datetime, '
                       'PULocationID, DOLocationID '
                       f'FROM {table_name} '
                       f'WHERE PULocationID < 264 '
                       f'AND DOLocationID < 264 '
                       f'LIMIT {block_size} '
                       f'OFFSET {blk_idx * block_size}')
            query_start = time()
            try:
                cursor.execute(command)
            except Error as e:
                print(e)

            rows = np.array(cursor.fetchall())
            if verbose:
                print(f">>> Time taken for query: {time() - query_start} seconds")

            preproc_start = time()
            if i == 0:
                features, outputs = get_naive_features(rows, coords, boros, datetime_onehot=datetime_onehot, 
                                        weekdays_onehot=weekdays_onehot, include_loc_ids=include_loc_ids)
                if verbose:
                    print(f">>> Time taken for preproc: {time() - preproc_start} seconds")
            else:
                features_sample, outputs_sample = get_naive_features(rows, coords, boros, datetime_onehot=datetime_onehot, 
                                                    weekdays_onehot=weekdays_onehot, include_loc_ids=include_loc_ids)
                if verbose:
                    print(f">>> Time taken for preproc: {time() - preproc_start} seconds")
                features = sparse.vstack([features, features_sample], format="csr")
                outputs = np.concatenate((outputs, outputs_sample))

        yield features, outputs


def extract_features(conn, table_name, variant='all', size=None, block_size=None, 
    datetime_onehot=True, weekdays_onehot=True, include_loc_ids=True):
    """Reads the data from the database and obtains the features

    :conn: connection object to the database
    :table_name: name of the table holding the rides data
    :variant: which type of variant to choose for extracting data
        Must be one out of
            - all : extracts features from all the data
            - random : uses a random batch of data from the db
            - batch: Extracts the features from a batch of data
            from the table of the database, without shuffling, and 
            returns a generator for it
    :size: the size of the batch of data
        (Used only if variant='random' or 'batch')
    :block_size: the size of blocks
        (Used only if variant='batch')
    :datetime_onehot: boolean for whether we want a onehot represnetation for
        date and time values, or a single index one
    :weekdays_onehot: boolean for whether we want a onehot represnetation for
        day of the week value, or a single index one
    :include_loc_ids: boolean for whether to include locIds as one-hot
        in the feature vectors, or not 
    :returns: a sparse csr_matrix containing the feature vectors
        and a numpy array containing the corresponding values
        of the travel time
    """
    if variant == 'all':
        print('Extracting features from all the data in {}'.format(table_name))
        features, outputs = extract_all_features(conn, table_name, datetime_onehot=datetime_onehot, 
                                weekdays_onehot=weekdays_onehot, include_loc_ids=include_loc_ids)

    elif variant == 'random':
        if not isinstance(size, int):
            print('Please provide an integer size for the random batch.')
        print('Extracting features from a random batch of data of size {} in {}'.format(size, table_name))
        features, outputs = extract_random_data_features(conn, table_name, size, datetime_onehot=datetime_onehot, 
                                weekdays_onehot=weekdays_onehot, include_loc_ids=include_loc_ids)

    elif variant == 'batch':
        if size is None:
            sys.exit("Please provide the size of the batch.")
        if block_size is None:
            sys.exit("Please provide an block_size value.")
        if size % block_size > 0:
            sys.exit("Please provide a batch size that is a multiple of block size.")
        print('Extracting features from a batch of data of size {} block_size in {}'.format(size, block_size, table_name))
        return extract_batch_features(conn, table_name, size, block_size, datetime_onehot=datetime_onehot, 
                    weekdays_onehot=weekdays_onehot, include_loc_ids=include_loc_ids,replace_blk=True, verbose=True)
    
    else:
        sys.exit("Type must be one of {'all', 'random', 'batch'}.")

    return features, outputs


if __name__ == "__main__":
    db_name = "rides.db" 
    con = create_connection(db_name)   
    # We have a total of 67302302 entries in the rides table 
    features_, outputs_ = extract_features(con, "rides", variant='random', size=10,
                                                datetime_onehot=False, 
                                                weekdays_onehot=False, 
                                                include_loc_ids=False)
    # for idx, (features_, outputs_) in enumerate(extract_features(con, "rides", variant='batch', size=100000, block_size=1000)):
        # print(f'Batch {idx}) features: {features_.shape}, outputs: {outputs_.shape}')
        # break
    # extract_all_coordinates(con, 'coordinates')
    print(features_.shape)

