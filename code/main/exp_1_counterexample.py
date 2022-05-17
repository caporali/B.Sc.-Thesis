# input from input_exp_1_counterexample
# 	a parametri NN
#		a.1 	nh := numero di hidden layer 
#		a.2 	sizeinput := taglia dell'input
#		a.3 	sizeoutput := taglia dell'output
#		a.4 	act := funzione di attivazione
#		a.5 	growth := parametro da cui dipende l'ampiezza degli hidden layer (parametro di crescita)
#		a.6 	ty := tipo di crescita dell'ampiezza degli hidden layers
# 	b altri parametri per la simulazione
#		b.7 	ntest := numero di reti neurali generate
#		b.8 	dist := distribuzione da usare per generare i parametri della NN
# 	c parametri per i plot	
# 		c.9 	nbins := numero di settori in cui suddividere ntest per il plot dell'istogramma
#		c.10 	tex := variabile che determina se i plot saranno visibili o in pgf (rispettivamente 0 e 1)
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
#		0.2.2 generazione sample_input e preparazione outputarray
#		0.2.3 loop di esecuzione delle ntest NNs
#			0.2.3.1 assegnazione parametri test-esima NN e scelta della distribuzione
#				0.2.3.1.1 esponenziale Exp(lambda_rate)
#				0.2.3.1.2 bernoulli B(p)
#				0.2.3.1.3 uniforme U(low, high)
#				0.2.3.1.4 normale N(0,C_A^(mu)) e N(0,C_b^(mu))
#			0.2.3.2 applicazione modello a test-esima NN
#			0.2.3.3 loading pretty print
#
# 1 test di gaussianità della NN nel limite dell'ampiezza degli hidden layers	
#	1.1 fit dell'output generato dalla NN
#	1.1 istogramma dell'output generato dalla NN
#


# -1

# -1.1
import sys
import os
import math
import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -1.2
sys.dont_write_bytecode = True

# -1.3
import input.input_exp_1_counterexample as inp

# -1.4
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath( __file__ ))))
from functions.tictoc import *
from functions.tensor_functions import *
from functions.nn_functions import *
from functions.other_functions import *

# -1.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -1.6
path = "plot/exp_1_counterexample/"
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
print("\t\t- n. generated NNs: " + str(inp.ntest) + "\n\t\t- n. sample input = " + str(1))

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
sample_input = 10*torch.rand(inp.sizeinput).to(device)
outputarray = np.zeros(inp.ntest)
print("\n0.2. generating NNs with " + inp.dist + " distribution:" + "\n\tcomputing...")

# 0.2.3
for test in range(inp.ntest):
	model = NeuralNetwork(inp.nh, inp.sizeinput, inp.sizeoutput, inp.growth, inp.ty).to(device)

	# 0.2.3.1
	i = 1
	bias_flag = 0

	# 0.2.3.1.1
	if inp.dist == "Exponential":
		lambda_rate = 0.5
		for param in model.parameters():
			if (bias_flag == 0):
				param.data = torch.tensor(np.random.exponential(scale = 1/lambda_rate, size = (param.size(0), param.size(1)))).float().to(device)
				bias_flag = 1		
			else:
				param.data = torch.tensor(np.random.exponential(scale = 1/lambda_rate, size = (param.size(0)))).float().to(device)
				bias_flag = 0
				i = i + 1

	# 0.2.3.1.2
	elif inp.dist == "Bernoulli":
		p = 0.5
		for param in model.parameters():
			if (bias_flag == 0):
				param.data = torch.tensor(np.random.binomial(1, 0.5, size = (param.size(0), param.size(1)))).float().to(device)
				bias_flag = 1		
			else:
				param.data = torch.tensor(np.random.binomial(1, 0.5, size = (param.size(0)))).float().to(device)
				bias_flag = 0
				i = i + 1

	# 0.2.3.1.3
	elif inp.dist == "Uniform":
		low = 0.
		high = 1.
		for param in model.parameters():
			if (bias_flag == 0):
				param.data = torch.tensor(np.random.uniform(low = 0, high = 1, size = (param.size(0), param.size(1)))).float().to(device)
				bias_flag = 1		
			else:
				param.data = torch.tensor(np.random.uniform(low = 0, high = 1, size = (param.size(0)))).float().to(device)
				bias_flag = 0
				i = i + 1

	# 0.2.3.1.4
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
				param.data = torch.normal(mean = torch.zeros(param.size(0), param.size(1)), std = sqrtca[i - 1]*torch.ones(param.size(0), param.size(1))).to(device)
				bias_flag = 1
			else:
				param.data = torch.normal(mean = torch.zeros(param.size(0)), std = sqrtcb*torch.ones(param.size(0))).to(device)
				bias_flag = 0
				i = i + 1

	# 0.2.3.2
	outputarray[test] = model(sample_input).cpu().detach().numpy()

	# 0.2.3.3
	if test%100 == 0:
		print("\t\t" + str(test) + "...")


# 1

# 1.1
mean = 0
std = 0
x = np.linspace(min(outputarray), max(outputarray), 250)
fitted = np.zeros(250)
mean, std = norm.fit(outputarray)
fitted = norm.pdf(x, mean, std)

# 1.2
print("\n1. plot: histogram of the NN valued on the sample_input")
plt.figure(0)
plt.hist(outputarray, bins = inp.nbins, color = "gray", density = True)
plt.plot(x, fitted, "k-") 
if inp.tex == 1:
	plt.title("\\texttt{sample_input}\n", fontsize = 10)
	plt.xlabel("NNs valued on \\texttt{sample_input}")
	plt.ylabel("density")
	plt.savefig(path + "exp_1_counterexample_" + inp.switch + ".pgf")
else:
	plt.title("sample_input\n", fontsize = 10)
	plt.xlabel("NNs valued on sample_input")
	plt.ylabel("density")
	plt.show()