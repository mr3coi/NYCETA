import numpy as np
from utils import create_connection
from obtain_features import *

super_boroughs = {
		'mbe': ['Manhattan', 'Bronx', 'EWR'],
		'bq': ['Brooklyn', 'Queens'],
		'si': ['Staten Island']
	}

for sb in super_boroughs:
	print(f"Extracting features for {super_boroughs[sb]}")
	con = create_connection('rides.db')

	features_, outputs_ = extract_features(con, "rides", variant='all', size=10, super_boro=super_boroughs[sb],
                                                datetime_onehot=False, 
                                                weekdays_onehot=False, 
                                                include_loc_ids=False)

	# the format of each feature vector
	#  [PUDatetime, PUCoords, DOCoords, PUBoroughs, DOBoroughs, PULocID, DOLocID]
	
	features_ = np.array(features_.toarray())	
	outputs_ = np.array(outputs_.toarray())

	np.save(features_, "features_"+sb+".npy")
	np.save(outputs_, "outputs_"+sb+".npy")