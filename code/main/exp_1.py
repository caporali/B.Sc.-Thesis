# input from input_exp_1
# 	a parametri NN
#		a.1 	nh := numero di hidden layer 
#		a.2 	sizeinput := taglia dell'input
#		a.3 	sizeoutput := taglia dell'output
#		a.4 	act := funzione di attivazione
#		a.5 	growth := parametro da cui dipende l'ampiezza degli hidden layer (parametro di crescita)
#		a.6 	ty := tipo di crescita dell'ampiezza degli hidden layers
# 	b altri parametri per la simulazione
#		b.7 	ntest := numero di reti neurali generate (tutte valutate negli nsample sample input)
#		b.8 	nsample := numero di sample input
# 	c parametri per i plot	
# 		c.9 	sample_0 := numero < (nsample - 1) associato al plot 1.3.1
# 		c.10 	nbins := numero di settori in cui suddividere ntest per il plot dell'istogramma
#		c.11 	tex := variabile che determina se i plot saranno visibili o in pgf (rispettivamente 0 e 1)
#
# -1 presets
#	-1.1 imported packages
#	-1.2 prevent cache creation
# 	-1.3 input
#	-1.4 set path to parent directory
#	-1.5 set graphic card 
#	-1.6 sets matplotlib
#
# 0 raccoglimento dei dati
#	0.1 stampa parametri
#	0.2 costruzione ed esecuzione delle NNs
#		0.2.1 costruzione modello NN		
#		0.2.2 \hat{C}_A^\mu, C_A^\mu e C_b^\mu
#		0.2.3 generazione sample_input e preparazione outputarray
#		0.2.4 loop di esecuzione delle ntest NNs
#			0.2.4.1 assegnazione parametri test-esima NN (in accordo con la teoria 0.2.2)
#			0.2.4.2 applicazione modello alla test-esima NN
#			0.2.4.3 loading pretty print
#
# 1 test sui dati raccolti 
#	1.1 test di gaussianità della NN nel limite dell'ampiezza degli hidden layers	
#		1.1.1 fit dell'output di sample_0 generato dalla NN
#		1.1.2 fit di una combinazione lineare degli output della NN
#	1.2 vettore delle medie e matrice delle covarianze
#	1.3 plot dei test
# 		1.3.1 fit ed istogramma per sample_input[sample_0]
#		1.3.2 fit ed istogramma per linear_combination
#


# -1

# -1.1
import sys
import os
import math
import numpy as np
import random
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -1.2
sys.dont_write_bytecode = True

# -1.3
import input.input_exp_1 as inp

# -1.4
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath( __file__ ))))
from functions.tictoc import *
from functions.tensor_functions import *
from functions.nn_functions import *
from functions.other_functions import *

# -1.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -1.6
path = "plot/exp_1/"
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
print("0. collecting data\n\n0.1. NN structure: \n\t\t- n. of hidden layers = " + str(inp.nh) + "\n\t\t- input size = " + str(inp.sizeinput))
print("\t\t- output size = " + str(inp.sizeoutput) + "\n\t\t- growth parameter = " + str(inp.growth) + "\n\t\t- type of growth = " + inp.ty)
print("\t\t- n. generated NNs: " + str(inp.ntest) + "\n\t\t- n. sample input = " + str(inp.nsample) + "\n\tcomputing...")

# 0.2

# 0.2.1
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

# 0.2.2
c = 0.8
ca = torch.ones(inp.nh + 1)
for j in range(inp.nh + 1): 
	ca[j] = c/hmu(j - 1, inp.growth, inp.ty) 
cb = 0.2
sqrtca = torch.sqrt(ca)
sqrtcb = math.sqrt(cb)

# 0.2.3
sample_input = torch.zeros(inp.nsample, inp.sizeinput).to(device)
for i in range(inp.nsample):
	sample_input[i, :] = 10*torch.rand(1, inp.sizeinput)
outputarray = np.zeros((inp.nsample, inp.ntest))

# 0.2.4
for test in range(inp.ntest):
	model = NeuralNetwork(inp.nh, inp.sizeinput, inp.sizeoutput, inp.growth, inp.ty).to(device)

	# 0.2.4.1
	i = 1
	bias_flag = 0
	for param in model.parameters():
		if (bias_flag == 0):
			param.data = torch.normal(mean = torch.zeros(param.size(0), param.size(1)), std = sqrtca[i - 1]*torch.ones(param.size(0), param.size(1))).to(device)
			bias_flag = 1		
		else:
			param.data = torch.normal(mean = torch.zeros(param.size(0)), std = sqrtcb*torch.ones(param.size(0))).to(device)
			bias_flag = 0
			i = i + 1

	# 0.2.4.2
	for i in range(inp.nsample):
		outputarray[i, test] = model(sample_input[i, :]).cpu().detach().numpy()

	# 0.2.4.3
	if test%100 == 0:
		print("\t\t" + str(test) + "...")


# 1

# 1.1

# 1.1.1
mean = np.zeros(inp.nsample)
std = np.zeros(inp.nsample)
x_0 = 1.2*max(-min(outputarray[inp.sample_0,:]), max(outputarray[inp.sample_0,:]))
x = np.linspace(-x_0, x_0, 250)
fitted = np.zeros((inp.nsample, 250))
for i in range(inp.nsample):
	mean[i], std[i] = norm.fit(outputarray[i,:])
	fitted[i, :] = norm.pdf(x, mean[i], std[i])

# 1.1.2
a = np.zeros(inp.nsample)
linear_combination = np.zeros_like(outputarray[0,:])
for i in range(inp.nsample):
	a[i] = random.randint(1, 10)
	linear_combination = linear_combination + a[i]*outputarray[i, :]
xl_0 = 1.2*max(-min(linear_combination), max(linear_combination))
xl = np.linspace(-xl_0, xl_0, 250)
meanl, stdl = norm.fit(linear_combination)
fittedl = norm.pdf(xl, meanl, stdl)

# 1.2
cov = np.zeros((inp.nsample, inp.nsample))
for i in range(inp.nsample):
	for j in range(inp.nsample):
		cov[i,j] = numerical_covariance(outputarray[i,:], outputarray[j,:]);
print("\n1. tests\n\n1.1. mean array and covariance matrix of the NNs randomly generated and evaluated on sample_input[i], for i = 0,...," + str(inp.nsample - 1) + ":")
print("mean:\n", mean, "\ncovariance:\n", cov)
toc()

# 1.3

# 1.3.1
print("\n1.2. plot: \n\t- histogram of the NNs valued on sample_input[" + str(inp.sample_0) + "]")
print("\t- N(" + str(round(mean[inp.sample_0], 3)) + "," + str(round(std[inp.sample_0]**2, 3)) + ") (fit of sample_input[" + str(inp.sample_0) + "])")
plt.figure(0)
plt.hist(outputarray[inp.sample_0, :], bins = inp.nbins, color = "gray", density = True)
plt.plot(x, fitted[inp.sample_0], "k-") 
if inp.tex == 1:
	strnormal = "$\\mathcal{N}\\left(" + str(round(mean[inp.sample_0], 3)) + ", \\," + str(round(std[inp.sample_0]**2, 3)) + "\\right)$"
	plt.title("\\texttt{sample_input[" + str(inp.sample_0) + "]}: " + strnormal + "\n", fontsize = 10)
	plt.xlabel("NNs valued on \\texttt{sample_input[" + str(inp.sample_0) + "]}")
	plt.ylabel(strnormal + " density")
	plt.savefig(path + "exp_1_sample_0.pgf")
else:
	strnormal = "N(" + str(round(mean[inp.sample_0], 3)) + ", " + str(round(std[inp.sample_0]**2, 3)) + ")"
	plt.title("sample_input[" + str(inp.sample_0) + "]: " + strnormal + "\n", fontsize = 10)
	plt.xlabel("NNs valued on sample_input[" + str(inp.sample_0) + "]")
	plt.ylabel(strnormal + " density")

# 1.3.2
print("\n1.3. plot: \n\t- histogram of the NNs valued on linear_combination\n\t  coefficients = ", a)
print("\t- N(" + str(round(meanl, 3)) + "," + str(round(stdl**2, 3)) + ") (fit of linear_combination)")
plt.figure(1)
plt.hist(linear_combination, bins = inp.nbins, color = "gray", density = True)
plt.plot(xl, fittedl, "k-")
if inp.tex == 1:
	strnormal = "$\\mathcal{N}\\left(" + str(round(meanl, 3)) + ", \\," + str(round(stdl**2, 3)) + "\\right)$"
	plt.title("\\texttt{linear_combination}: " + strnormal + "\n", fontsize = 10)
	plt.xlabel("\\texttt{linear_combination} obtained from previous NNs")
	plt.ylabel(strnormal + " density")
	plt.savefig(path + "exp_1_linear_combination.pgf")
else:
	strnormal = "N(" + str(round(meanl, 3)) + ", " + str(round(stdl**2, 3)) + ")"
	plt.title("linear_combination: " + strnormal + "\n", fontsize = 10)
	plt.xlabel("linear_combination obtained from previous NNs")
	plt.ylabel(strnormal + " density")
	plt.show()