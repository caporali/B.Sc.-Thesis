# -1 presets
#	-1.1 imported packages
#
# 0 input
#


# -1

# -1.1
import numpy as np


# 0

switch = "1.10"

if switch == "0.04":
	
	# a
	fstr = "5x^4 - 2x^2 - x + 7" 
	def f(x):
		return  5*x**4 - 2*x**2 - x + 7 
	a, b = -4, 4
	nsample = 100
	# b
	nh = 4
	sizeinput = 1
	sizeoutput = 1
	act = "ReLU"
	growth = 100
	ty = "const"
	# c
	nepoch = 200
	err = 125
	chosen_lr = 3.5*1e-7

elif switch == "0.06":
	
	# a
	fstr = "5x^4 - 2x^2 - x + 7" 
	def f(x):
			return  5*x**4 - 2*x**2 - x + 7 
	a, b = -4, 4
	nsample = 100
	# b
	nh = 6
	sizeinput = 1
	sizeoutput = 1
	act = "ReLU"
	growth = 100
	ty = "const"
	# c
	nepoch = 200
	err = 125
	chosen_lr = 3.5*1e-7

elif switch == "r1_0.06":
	
	# a
	fstr = "5x^4 - 2x^2 - x + 7" 
	def f(x):
			return  5*x**4 - 2*x**2 - x + 7 
	a, b = -4, 4
	nsample = 100
	# b
	nh = 6
	sizeinput = 1
	sizeoutput = 1
	act = "ReLU1"
	growth = 100
	ty = "const"
	# c
	nepoch = 200
	err = 500
	chosen_lr = 5*1e-7

elif switch == "1.01":
	
	# a
	fstr = "cos(x^2) - sin(2x + 3)"
	def f(x):
			return np.cos(x**2) - np.sin(2*x + 3)
	a, b = -2, 2
	nsample = 100
	# b
	nh = 1
	sizeinput = 1
	sizeoutput = 1
	act = "ReLU1"
	growth = 100
	ty = "const"
	# c
	nepoch = 200
	err = 0.3
	chosen_lr = 4*1e-3

elif switch == "1.02":
	
	# a
	fstr = "cos(x^2) - sin(2x + 3)"
	def f(x):
			return np.cos(x**2) - np.sin(2*x + 3)
	a, b = -2, 2
	nsample = 100
	# b
	nh = 2
	sizeinput = 1
	sizeoutput = 1
	act = "ReLU1"
	growth = 100
	ty = "const"
	# c
	nepoch = 200
	err = 0.25
	chosen_lr = 4*1e-3

elif switch == "1.04":
	
	# a
	fstr = "cos(x^2) - sin(2x + 3)"
	def f(x):
			return np.cos(x**2) - np.sin(2*x + 3)
	a, b = -2, 2
	nsample = 100
	# b
	nh = 4
	sizeinput = 1
	sizeoutput = 1
	act = "ReLU1"
	growth = 100
	ty = "const"
	# c
	nepoch = 200
	err = 0.20
	chosen_lr = 4*1e-3

elif switch == "1.10":
	
	# a
	fstr = "cos(x^2) - sin(2x + 3)"
	def f(x):
			return np.cos(x**2) - np.sin(2*x + 3)
	a, b = -2, 2
	nsample = 100
	# b
	nh = 10
	sizeinput = 1
	sizeoutput = 1
	act = "ReLU1"
	growth = 100
	ty = "const"
	# c
	nepoch = 500
	err = 0.15
	chosen_lr = 5.5*1e-3

elif switch == "s_2.02":
	
	# a
	fstr = "|x| e^x"
	def f(x):
			return np.absolute(x)*np.exp(x)
	a, b = -1, 1
	nsample = 50
	# b
	nh = 2
	sizeinput = 1
	sizeoutput = 1
	act = "Sigmoid"
	growth = 100
	ty = "const"
	# c
	nepoch = 50
	err = 0.2
	chosen_lr = 5*1e-2

elif switch == "s_2.04":
	
	# a
	fstr = "|x| e^x"
	def f(x):
			return np.absolute(x)*np.exp(x)
	a, b = -1, 1
	nsample = 50
	# b
	nh = 4
	sizeinput = 1
	sizeoutput = 1
	act = "Sigmoid"
	growth = 100
	ty = "const"
	# c
	nepoch = 1500
	err = 0.2
	chosen_lr = 6*1e-2

elif switch == "r1_2.02":
	
	# a
	fstr = "|x| e^x"
	def f(x):
			return np.absolute(x)*np.exp(x)
	a, b = -1, 1
	nsample = 50
	# b
	nh = 2
	sizeinput = 1
	sizeoutput = 1
	act = "ReLU1"
	growth = 100
	ty = "const"
	# c
	nepoch = 30
	err = 0.1
	chosen_lr = 1*1e-2

elif switch == "r1_2.04":
	
	# a
	fstr = "|x| e^x"
	def f(x):
			return np.absolute(x)*np.exp(x)
	a, b = -1, 1
	nsample = 50
	# b
	nh = 4
	sizeinput = 1
	sizeoutput = 1
	act = "ReLU1"
	growth = 100
	ty = "const"
	# c
	nepoch = 200
	err = 0.02
	chosen_lr = 1*1e-2

else:
	print("error: non valid input")
	quit()
	
# d
loss_track = 1
tex = 0
act_print = 1