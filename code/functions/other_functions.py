# -1 presets
#	-1.1 imported packages
#
# 0 list of other function
#	0.1 numerical_covariance
#


# -1

# -1.1
import numpy as np


# 0

# 0.1
def numerical_covariance(arr_1, arr_2):
	# numerical_covariance() := calcola la covarianza numerica tra due np array (population covariance)
	#
	#	input:
	#		arr_1 := primo np array
	#		arr_2 := secondo np array
	#
	#	output:
	#		numerical_covariance(arr_1, arr_2)) := Cov(arr_1, arr_2)
	#
	return np.cov(arr_1, arr_2, bias = True)[0][1]