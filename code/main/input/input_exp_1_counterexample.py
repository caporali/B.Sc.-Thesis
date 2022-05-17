# 0 input
#	0.1 counterexample activation
#	0.2 counterexample distribution 
# 


# 0

switch = "distribution"

if switch == "distribution":

	# 0.1

	# a
	nh = 2
	sizeinput = 4
	sizeoutput = 1
	act = "ReLU"
	growth = 100
	ty = "const"
	# b
	ntest = 10000
	dist = "Exponential"
	# c
	nbins = 45
	tex = 0

elif switch == "activation":

	#0.2

	# a
	nh = 2
	sizeinput = 4
	sizeoutput = 1
	act = "PRLU3"
	growth = 100
	ty = "const"
	# b
	ntest = 10000
	dist = "Normal"
	# c
	nbins = 60
	tex = 0