import numpy as np
from utils import create_connection
from obtain_features import *

super_boroughs = {
		# 'mbe': ['Manhattan', 'Bronx', 'EWR'],
		'si': ['Staten Island'],
		'bq': ['Brooklyn', 'Queens']
	}

for sb in super_boroughs:
	print(f"Extracting features for {super_boroughs[sb]}")
	con = create_connection('/home/ubuntu/data/rides.db')

	features_, outputs_ = extract_features(con, "rides", variant='all', size=10, super_boro=super_boroughs[sb],
                                                datetime_onehot=True, 
                                                weekdays_onehot=True, 
                                                include_loc_ids=True)

	# the format of each feature vector
	#  [PUDatetime, PUCoords, DOCoords, PUBoroughs, DOBoroughs, PULocID, DOLocID]
	
	features_ = np.array(features_.toarray())	
	outputs_ = np.array(outputs_)

	np.save(features_, "features_"+sb+".npy")
	np.save(outputs_, "outputs_"+sb+".npy")