# input from input_torch_test
#	 a parametri NN 																						
#		a.1 	nh := numero di hidden layer 
#		a.2 	sizeinput := taglia dell'input
#		a.3 	sizeoutput := taglia dell'output
#		a.4 	act := funzione di attivazione
#		a.5 	growth := parametro da cui dipende l'ampiezza degli hidden layer (parametro di crescita)
#		a.6 	ty := tipo di crescita dell'ampiezza degli hidden layers
# 	b parametri per plot
#		b.7 	tex := variabile che determina se i plot saranno visibili o in pgf (rispettivamente 0 e 1)
#
# -1 presets
#	-1.1 imported packages
#	-1.2 prevent cache creation
#	-1.3 input
#	-1.4 set path to parent directory
#	-1.5 set graphic card 
#	-1.6 sets matplotlib
#
# 0 definizione e costruzione della NN
#	0.1 costruzione NN
#	0.2 stampa NN
#
# 1. assegnazione dei parametri 
#	1.1	\hat{C}_A^\mu, C_A^\mu e C_b^\mu 
#		(i nomi delle variabili sono quelli presenti nel capitolo 3 della tesi ed i valori usati sono quelli riportati nella sezione 4 dell'articolo di Matthews)
#		- c :=  \hat{C}_A^\mu
#		- ca[mu] := C_A^\mu
#		- cb := C_b^\mu
#	1.2 assegnazione parametri della NN e pretty print
#		1.2.1 generazione parametri matrici N(0,C_A^(mu))
#		1.2.2 generazione parametri bias N(0,C_b^(mu))
#
# 2 sample: input-output
#
# 3 plot NN function
#	3.1 computation
#	3.2 plot
#


# -1

# -1.1
import sys
import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -1.2
sys.dont_write_bytecode = True

# -1.3
import input.input_torch_test as inp

# -1.4
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath( __file__ ))))
from functions.tictoc import *
from functions.tensor_functions import *
from functions.nn_functions import *
from functions.other_functions import *

# -1.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -1.6
path = "plot/torch_test/"
if inp.tex == 1:
	matplotlib.use("pgf")
	plt.rcParams.update({
		"pgf.texsystem": "pdflatex",
	    "font.family": "serif",
	    "text.usetex": True,
	    "pgf.rcfonts": False,
		"font.size": 8,
		"axes.axisbelow": True
	})
else:
	plt.rcParams.update({ 
	    "font.sans-serif": "Courier New",
		"font.size": 8,
		"axes.axisbelow": True
	})


# 0

tic()

# 0.1
class NeuralNetwork(nn.Module):
	def __init__(self, nh, sizeinput, sizeoutput, n, ty):
		super(NeuralNetwork, self).__init__()
		self.linears = nn.ModuleList([nn.Linear(sizeinput, hmu(1, n, ty))])
		for i in range(1, nh):
			self.linears.append(nn.Linear(hmu(i, n, ty), hmu(i + 1, n, ty)))
		self.linears.append(nn.Linear(hmu(nh, n, ty), sizeoutput, False))
	def forward(self, x):
		x = self.linears[0](x)
		for i in range(1, inp.nh + 1):
			x = activation(x, inp.act)
			x = self.linears[i](x)
		return x
model = NeuralNetwork(inp.nh, inp.sizeinput, inp.sizeoutput, inp.growth, inp.ty).to(device)

# 0.2
print("0. NN structure")
print("- activation: " + inp.act + "\n-", model, "\n")


# 1

# 1.1
c = 0.8
ca = torch.ones(inp.nh + 1)
for j in range(inp.nh + 1): 
	ca[j] = c/hmu(j - 1, inp.growth, inp.ty) 
cb = 0.2
sqrtca = torch.sqrt(ca)
sqrtcb = math.sqrt(cb)

# 1.2
print("1. NN parameters")
i = 1
bias_flag = 0
for param in model.parameters():
	if (bias_flag == 0):
		
		# 1.2.1
		print("W_" + str(i) + ":\n- size: " + str(param.size(0)) + "x" + str(param.size(1)) + "\n- distribution: N(0, " + str(round(ca[i - 1].item(), 2)) + ")")
		param.data = torch.normal(mean = torch.zeros(param.size(0), param.size(1)), std = sqrtca[i - 1]*torch.ones(param.size(0), param.size(1))).to(device)
		bias_flag = 1
	else:

		# 1.2.2
		print("b_" + str(i) + ":\n- size: " + str(param.size(0)) + "\n- distribution: N(0, " + str(round(cb, 2)) + ")")
		param.data = torch.normal(mean = torch.zeros(param.size(0)), std = sqrtcb*torch.ones(param.size(0))).to(device)
		bias_flag = 0
		i = i + 1
	print("- content: \n", print_tensor(param), "\n\t----")


# 2

print("\n2. test:")
sampleinput = torch.from_numpy(np.arange(1.0, inp.sizeinput + 1)).float().to(device);
print("\t- input: ", print_tensor(sampleinput))
sampleoutput = model(sampleinput)
print("\t- output:", print_tensor(sampleoutput))


# 3

if inp.sizeinput == 1 and inp.sizeoutput == 1:
	print("\n3. plot model function")
	
	# 3.1
	a = -5
	b = 5
	num = 400
	input_tensor = np.linspace(a, b, num = num)
	output_tensor = np.zeros_like(input_tensor)
	for i in range(num):
		output_tensor[i] = model(cast_tensor(input_tensor[i])).item()

	toc()	

	# 3.2
	plot_activation(inp.act, inp.tex, path)
	plt.figure(0)
	plt.plot(input_tensor, output_tensor, "k-")
	plt.xlim(a, b)	
	plt.grid()
	plt.gca().set_aspect(aspect = "auto", adjustable = "datalim") # plt.gca().set_aspect(aspect = "equal", adjustable = "datalim")
	if inp.tex == 1:
		plt.xlabel("$x$ (input)")
		plt.ylabel("model$(x)$ (model output)")
		plt.savefig(path + "torch_test_NN_" + inp.act + ".pgf")
	else:
		plt.xlabel("x (input)")
		plt.ylabel("model(x) (model output)")
		plt.show()