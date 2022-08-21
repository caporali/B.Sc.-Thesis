# -1 presets
#	-1.1 imported packages
#	-1.2 prevent cache creation
#	-1.3 input (gli stessi di exp_2)
#	-1.4 set path to parent directory
#	-1.5 set graphic card 
#	-1.6 sets matplotlib
#
# 0 inizializzazione e caricamento NN
# 	0.1 definizione NN come in exp_2 (stessi parametri del modello salvato)
#	0.2 caricamento NN 
#
# 1 tests
# 	1.1 applicazione della NN a x (x \in [a,b])
#	1.2 plot funzione generata dalla NN


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
import torch.optim as optim

# -1.2
sys.dont_write_bytecode = True

# -1.3 
import input.input_exp_2 as inp

# -1.4
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath( __file__ ))))
from functions.tictoc import *
from functions.tensor_functions import *
from functions.nn_functions import *
from functions.other_functions import *

# -1.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -1.6
path = "plot/exp_2/"
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

print("0. collecting data of saved NN\n\t- approximated function:\n\t\t- f(x) = " + inp.fstr + " \n\t\t- range = [" + str(inp.a) + ","  + str(inp.b) + "]")
print("\t\t- n. samples = " + str(inp.nsample) + "\n\t" + "- NN structure:\n\t\t- n. di hidden layers = " + str(inp.nh) + "\n\t\t- input size = " + str(inp.sizeinput))
print("\t\t- output size = " + str(inp.sizeoutput) + "\n\t\t- activation function = " + inp.act + "\n\t\t- growth parameter = " + str(inp.growth) + "\n\t\t- type of growth = " + inp.ty)

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
model.load_state_dict(torch.load("model.pt"))
model.eval()


# 1

# 1.1
num = int((inp.b - inp.a)*50)
x = np.linspace(inp.a, inp.b, num)
modelfx = np.zeros_like(x)
for var in range(x.size):
	modelfx[var] = model(cast_tensor(x[var])).item()

# 1.2
print("\n1. plot of saved NN: modelf(x)")
plt.plot(x, modelfx, "k-")
plt.xlim(inp.a, inp.b)
plt.grid()
if inp.tex == 1:
	plt.title("approximation of $f(x) = " + inp.fstr + "$ with NNs: modelf$(x)$\n", fontsize = 10)
	plt.xlabel("x")
	plt.ylabel("modelf$(x)$")
	plt.savefig(path + "exp_2_load.pgf")
else:
	plt.title("approximation of f(x) = " + inp.fstr + " with NNs: modelf(x)\n", fontsize = 10)
	plt.xlabel("x")
	plt.ylabel("modelf(x)")
	plt.show()
