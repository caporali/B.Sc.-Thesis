# input from input_torch_test_counterexample
#	 a parametri NN 																						
#		a.1 	nh := numero di hidden layer 
#		a.2 	sizeinput := taglia dell'input
#		a.3 	sizeoutput := taglia dell'output
#		a.4 	act := funzione di attivazione
#		a.5 	growth := parametro da cui dipende l'ampiezza degli hidden layer (parametro di crescita)
#		a.6 	ty := tipo di crescita dell'ampiezza degli hidden layers
# 	b altri parametri per la simulazione
#		b.7		dist := distribuzione da usare per generare i parametri della NN
#
# -1 presets
#	-1.1 imported packages
#	-1.2 prevent cache creation
#	-1.3 input
#	-1.4 set path to parent directory
#	-1.5 set graphic card 
#
# 0 definizione e costruzione della NN
#	0.1 costruzione NN
#	0.2 stampa NN
#
# 1. assegnazione parametri della NN e scelta della distribuzione
#	1.1 esponenziale Exp(lambda_rate)
#	1.2 bernoulli B(p)
#	1.3 uniforme U(low, high)
#	1.4 normale N(0,C_A^(mu)) e N(0,C_b^(mu))
#
# 2 sample: input-output


# -1

# -1.1
import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn

# -1.2
sys.dont_write_bytecode = True

# -1.3
import input.input_torch_test_counterexample as inp

# -1.4
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath( __file__ ))))
from functions.tictoc import *
from functions.tensor_functions import *
from functions.nn_functions import *
from functions.other_functions import *

# -1.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

print("1. NN parameters")
i = 1
bias_flag = 0

# 1.1
if inp.dist == "Exponential":
	lambda_rate = 1.5
	for param in model.parameters():
		if (bias_flag == 0):
			print("W_" + str(i) + ":\n- size: " + str(param.size(0)) + "x" + str(param.size(1)) + "\n- distribution: Exp(" + str(lambda_rate) + ")")
			param.data = torch.tensor(np.random.exponential(scale = 1/lambda_rate, size = (param.size(0), param.size(1)))).float().to(device)
			bias_flag = 1		
		else:
			print("b_" + str(i) + ":\n- size: " + str(param.size(0)) + "\n- distribution: Exp(" + str(lambda_rate) + ")")
			param.data = torch.tensor(np.random.exponential(scale = 1/lambda_rate, size = (param.size(0)))).float().to(device)
			bias_flag = 0
			i = i + 1
		print("- content: \n", print_tensor(param), "\n\t----")

# 1.2
elif inp.dist == "Bernoulli":
	p = 0.5
	for param in model.parameters():
		if (bias_flag == 0):
			print("W_" + str(i) + ":\n- size: " + str(param.size(0)) + "x" + str(param.size(1)) + "\n- distribution: B(" + str(p) + ")")
			param.data = torch.tensor(np.random.binomial(1, 0.5, size = (param.size(0), param.size(1)))).float().to(device)
			bias_flag = 1		
		else:
			print("b_" + str(i) + ":\n- size: " + str(param.size(0)) + "\n- distribution: B(" + str(p) + ")")
			param.data = torch.tensor(np.random.binomial(1, 0.5, size = (param.size(0)))).float().to(device)
			bias_flag = 0
			i = i + 1
		print("- content: \n", print_tensor(param), "\n\t----")

# 1.3
elif inp.dist == "Uniform":
	low = 0.
	high = 1.
	for param in model.parameters():
		if (bias_flag == 0):
			print("W_" + str(i) + ":\n- size: " + str(param.size(0)) + "x" + str(param.size(1)) + "\n- distribution: U(" + str(low) + "," + str(high) + ")")
			param.data = torch.tensor(np.random.uniform(low = 0, high = 1, size = (param.size(0), param.size(1)))).float().to(device)
			bias_flag = 1		
		else:
			print("b_" + str(i) + ":\n- size: " + str(param.size(0)) + "\n- distribution: U(" + str(low) + "," + str(high) + ")")
			param.data = torch.tensor(np.random.uniform(low = 0, high = 1, size = (param.size(0)))).float().to(device)
			bias_flag = 0
			i = i + 1
		print("- content: \n", print_tensor(param), "\n\t----")

# 1.4
elif inp.dist == "Normal":
	c = 0.8
	ca = torch.ones(inp.nh + 1)
	for j in range(inp.nh + 1): 
		ca[j] = c/hmu(j - 1, inp.growth, inp.ty) 
	cb = 0.2
	sqrtca = torch.sqrt(ca)
	sqrtcb = math.sqrt(cb)
	for param in model.parameters():
		if (bias_flag == 0):
			print("W_" + str(i) + ":\n- size: " + str(param.size(0)) + "x" + str(param.size(1)) + "\n- distribution: N(0, " + str(round(ca[i - 1].item(), 2)) + ")")
			param.data = torch.normal(mean = torch.zeros(param.size(0), param.size(1)), std = sqrtca[i - 1]*torch.ones(param.size(0), param.size(1))).to(device)
			bias_flag = 1
		else:
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