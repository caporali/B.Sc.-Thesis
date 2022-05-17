# 0 input
#	0.1 variable input
#	0.2 input for visible gaussian
#


# 0

switch = "normal"

if switch == "normal":

	# a
	nh = 1
	sizeinput = 3
	sizeoutput = 2
	act = "ReLU1"
	maxgrowth = 1000
	stepgrowth = int(maxgrowth)
	ty = "const"
	# b
	ntest = 1000
	# c
	nbins = 30
	tex = 0

elif switch == "gaussian":

	# a
	nh = 1
	sizeinput = 3
	sizeoutput = 2
	act = "ReLU1"
	maxgrowth = 100000
	stepgrowth = int(maxgrowth)
	ty = "const"
	# b
	ntest = 100000
	# c
	nbins = 30
	tex = 1