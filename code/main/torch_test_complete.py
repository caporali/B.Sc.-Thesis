# input from input_torch_test_complete
#	 a parametri NN 																						
#		a.1 	sizeinput := taglia dell'input
#		a.2 	sizeoutput := taglia dell'output
#		a.3 	growth := parametro da cui dipende l'ampiezza degli hidden layer (parametro di crescita)
#		a.4 	ty := tipo di crescita dell'ampiezza degli hidden layers
# 	b parametri per plot
#		b.5 	tex := variabile che determina se i plot saranno visibili o in pgf (rispettivamente 0 e 1)
#
# -1 presets
#	-1.1 imported packages
#	-1.2 prevent cache creation
#	-1.3 input
#	-1.4 set path to parent directory
#	-1.5 set graphic card 
#	-1.6 sets matplotlib
#
# 0 definizione e costruzione delle NNs
#	0.1 costruzione model (usando nn.Sequential)
#	0.2 costruzione model_par (parametrica)
#	0.3 stampa NNs
#		0.3.1 model
#		0.3.2 model_par
#
# 1. assegnazione dei parametri 
#	1.1	\hat{C}_A^\mu, C_A^\mu e C_b^\mu 
#		(i nomi delle variabili sono quelli presenti nel capitolo 3 della tesi ed i valori usati sono quelli riportati nella sezione 4 dell'articolo di Matthews)
#		- c :=  \hat{C}_A^\mu
#		- ca[mu] := C_A^\mu
#		- cb := C_b^\mu
#	1.2 assegnazione parametri della NN e pretty print (i commenti etichettati con ~ sono test di coerenza dei dati trovati)
#		1.2.1 generazione parametri matrici N(0,C_A^(mu))
#		1.2.2 generazione parametri bias N(0,C_b^(mu))
#		1.2.3 assegnazione medesimi parametri a model_par
#		1.2.4 pretty print
#
# 2 sample: input-output
#
# 3 plot NN function
#	3.1 computation
#	3.2 plot
#	3.3 plot (parametric version)
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
import input.input_torch_test_complete as inp

# -1.4
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath( __file__ ))))
from functions.tictoc import *
from functions.tensor_functions import *
from functions.nn_functions import *
from functions.other_functions import *

# -1.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -1.6
path = "plot/torch_test_complete/"
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

nh = 4

# 0.1
class NeuralNetwork(nn.Module):
	def __init__(self, nh, sizeinput, sizeoutput, n, ty):
		super(NeuralNetwork, self).__init__()
		self.linear_activation_stack = nn.Sequential(
			nn.Linear(sizeinput, hmu(1, n, ty)),
			nn.Sigmoid(),
			nn.Linear(hmu(1, n, ty), hmu(2, n, ty)),
			nn.Sigmoid(),
			nn.Linear(hmu(2, n, ty), hmu(3, n, ty)),
			nn.Sigmoid(),
			nn.Linear(hmu(3, n, ty), hmu(4, n, ty)),
			nn.Sigmoid(),
			nn.Linear(hmu(4, n, ty), sizeoutput, False),
		)
	def forward(self, x):
		return self.linear_activation_stack(x)
model = NeuralNetwork(nh, inp.sizeinput, inp.sizeoutput, inp.growth, inp.ty).to(device)

# 0.1_alternative
# class NeuralNetwork(nn.Module):
# 	def __init__(self, nh, sizeinput, sizeoutput, n, ty):
# 		super(NeuralNetwork, self).__init__()
# 		self.linear1 = nn.Linear(sizeinput, hmu(1, n, ty))
# 		self.linear2 = nn.Linear(hmu(1, n, ty), hmu(2, n, ty))
# 		self.linear3 = nn.Linear(hmu(2, n, ty), hmu(3, n, ty))
# 		self.linear4 = nn.Linear(hmu(3, n, ty), hmu(4, n, ty))
# 		self.linear5 = nn.Linear(hmu(4, n, ty), sizeoutput, False)
# 	def forward(self, x):
# 		# x = in
# 		p_h1 = self.linear1(x)				
# 		h1 = activation(p_h1, act)		
# 		p_h2 = self.linear2(h1)				
# 		h2 = activation(p_h2, act)		
# 		p_h3 = self.linear3(h2)				
# 		h3 = activation(p_h3, act)		
# 		p_h4 = self.linear4(h3)				
# 		h4 = activation(p_h4, act)		
# 		out = self.linear5(h4)				
# 		return out
# model = NeuralNetwork(nh, inp.sizeinput, inp.sizeoutput, inp.growth, inp.ty).to(device)

# 0.2
act = "Sigmoid"
class NeuralNetwork(nn.Module):
	def __init__(self, nh, sizeinput, sizeoutput, n, ty):
		super(NeuralNetwork, self).__init__()
		self.linears = nn.ModuleList([nn.Linear(sizeinput, hmu(1, n, ty))])
		for i in range(1, nh):
			self.linears.append(nn.Linear(hmu(i, n, ty), hmu(i + 1, n, ty)))
		self.linears.append(nn.Linear(hmu(nh, n, ty), sizeoutput, False))
	def forward(self, x):
		x = self.linears[0](x)
		for i in range(1, nh + 1):
			x = activation(x, act)
			x = self.linears[i](x)
		return x
model_par = NeuralNetwork(nh, inp.sizeinput, inp.sizeoutput, inp.growth, inp.ty).to(device)

# 0.3

# 0.3.1
print("0. NNs structures\n\n0.1 model structure")
print("-", model, "\n")

# 0.3.2
print("0.2 model_par structure")
print("- activation: " + act + "\n-", model_par, "\n")


# 1

# 1.1
c = 0.8
ca = torch.ones(nh + 1)
for j in range(nh + 1): 
	ca[j] = c/hmu(j - 1, inp.growth, inp.ty) 
cb = 0.2
sqrtca = torch.sqrt(ca)
sqrtcb = math.sqrt(cb)

# 1.2
print("1. NNs\n\n1.1 model parameters")
j = 1
i = 1
bias_flag = 0
for param in model.parameters():
	if (bias_flag == 0):
		
		# 1.2.1
		print("W_" + str(i) + ":\n- size: " + str(param.size(0)) + "x" + str(param.size(1)) + "\n- distribution: N(0, " + str(round(ca[i - 1].item(), 2)) + ")")
		param.data = torch.normal(mean = torch.zeros(param.size(0), param.size(1)), std = sqrtca[i - 1]*torch.ones(param.size(0), param.size(1))).to(device)
		bias_flag = 1
		# ~
		# 
		#	std = 0
		#	for k in param: 
		#		for l in k:
		# 			std = std + l.item()**2
		#	std = math.sqrt(std/(param.size(0) * param.size(1) - 1))
		#	print("- data check \n	- real std:" + str(round(sqrtca[i - 1].item(), 2)) + "\n	- data std:" + str(std))	 
		#
	else:

		# 1.2.2
		print("b_" + str(i) + ":\n- size: " + str(param.size(0)) + "\n- distribution: N(0, " + str(round(cb, 2)) + ")")
		param.data = torch.normal(mean = torch.zeros(param.size(0)), std = sqrtcb*torch.ones(param.size(0))).to(device)
		bias_flag = 0
		i = i + 1
		# ~
		# 
		#	std = 0
		#	for k in param: 
		#		std = std + k.item()**2
		#	std = math.sqrt(std/(param.size(0) - 1))
		#	print("- data check \n	- real std:" + str(round(sqrtcb, 2)) + "\n	- data std:" + str(std))
		#
	
	# 1.2.3
	k = 1
	for param_par in model_par.parameters():
		if k == j:
			param_par.data = param.data
		k = k + 1 
	j = j + 1
	
	# 1.2.4
	print("- content: \n", print_tensor(param), "\n\t----")

# 1.3
print("\n1.2. model_par parameters")
i = 1
bias_flag = 0
for param in model_par.parameters():
	if (bias_flag == 0):
		print("W_" + str(i) + ":\n- size: " + str(param.size(0)) + "x" + str(param.size(1)) + "\n- distribution: N(0, " + str(round(ca[i - 1].item(), 2)) + ")")
		bias_flag = 1
	else:
		print("b_" + str(i) + ":\n- size: " + str(param.size(0)) + "\n- distribution: N(0, " + str(round(cb, 2)) + ")")
		bias_flag = 0
		i = i + 1
	print("- content: \n", print_tensor(param), "\n\t----")


#2

print("\n2. test:")
sampleinput = torch.from_numpy(np.arange(1.0, inp.sizeinput + 1)).float().to(device);
print("\t- input: ", print_tensor(sampleinput))
sampleoutput = model(sampleinput)
print("\t- output:", print_tensor(sampleoutput))
sampleoutput_par = model(sampleinput)
print("\t- output_par:", print_tensor(sampleoutput_par))


#3

if inp.sizeinput == 1:
	print("\n3. plot model functions")
	
	#3.1
	a = -20 
	b = 20
	num = 400
	input_tensor = np.linspace(a, b, num = num)
	output_tensor = np.zeros_like(input_tensor)
	output_tensor_par = np.zeros_like(input_tensor)
	for i in range(num):
		output_tensor[i] = model(cast_tensor(input_tensor[i])).item()
		output_tensor_par[i] = model_par(cast_tensor(input_tensor[i])).item()

	toc()	

	# 3.2
	plt.figure(0)
	plt.plot(input_tensor, output_tensor, "r-", label = "model")
	plt.plot(input_tensor, output_tensor_par, "k--",  label = "model_par")
	plt.xlim(a, b)
	plt.grid()
	plt.legend()
	if inp.tex == 1:
		plt.xlabel("$x$ (input)")
		plt.ylabel("model$(x)$ / model_par$(x)$ (model output)")
		plt.savefig(path + "torch_test_complete_NN.pgf")
	else:
		plt.xlabel("x (input)")
		plt.ylabel("model(x) (model output) / model_par(x)")
		plt.show()