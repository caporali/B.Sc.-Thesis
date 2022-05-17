# -1 presets
#	-1.1 imported packages
#
# 0 definition of tictoc function
#	0.1 generator [2 parameters:
#		- ti := initial time;
#		- tf := final time;]
#	0.2 tic := records a time in TicToc, marks the beginning of a time interval
# 	0.3 toc :=  prints the time difference yielded by generator instance TicToc
#


# -1

# -1.1
import time


# 0

# 0.1
def TicToc_Generator():
	ti = 0
	tf = time.time()
	while True:
		ti = tf
		tf = time.time()
		yield tf - ti
TicToc = TicToc_Generator() 

# 0.2
def tic():	
	toc(False)

# 0.3
def toc(temp_bool = True):
	temp_time_interval = next(TicToc)
	if temp_bool:
		print("\ntime: %f seconds" % temp_time_interval)