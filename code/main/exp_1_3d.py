# input from input_exp_1_3d
# 	a parametri NN
#		a.1 	nh := numero di hidden layer 
#		a.2 	sizeinput := taglia dell'input
#		a.3 	sizeoutput := taglia dell'output
#		a.4 	act := funzione di attivazione
#		a.5		maxgrowth := massimo valore del parametro da cui dipende l'ampiezza degli hidden layer (parametro di crescita) 
#		a.6 	stepgrowth := valore della distanza tra ciascuna coppia di valori growth usati
#				(maxgrowth = 1000, stepgrowth = 200 allora growth = 200, 400, 600, 800, 1000)
#		a.7 	ty := tipo di crescita dell'ampiezza degli hidden layers
# 	b altri parametri per la simulazione
#		b.8 	ntest := numero di reti neurali generate
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
#		0.2.3 loop di scorrimento delle steps reti neurali
#			0.2.3.1 pretty print della corrente growth
#			0.2.3.2 \hat{C}_A^\mu, C_A^\mu e C_b^\mu
#			0.2.3.3 loop di esecuzione delle ntest NNs associate al corrente parametro growth
#				0.2.3.3.1 assegnazione parametri test-esima NN (in accordo con la teoria 0.2.3.2)
#				0.2.3.3.2 applicazione modello alla test-esima NN
#				0.2.3.3.3 loading pretty print
# 			0.2.3.4 test sui dati raccolti (corrente growth)
# 				0.2.3.4.1 computo dei parametri necessari ai plot a partire da outputarray
#					0.2.3.4.1.1 istogramma 3d della densità della NN con il corrente parametro growth 
#					0.2.3.4.1.2 tri-surface plot della densità della NN con il corrente parametro growth 
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
import input.input_exp_1_3d as inp

# -1.4
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath( __file__ ))))
from functions.tictoc import *
from functions.tensor_functions import *
from functions.nn_functions import *
from functions.other_functions import *

# -1.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -1.6
path = "plot/exp_1_3d/"
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
print("\t\t- output size = " + str(inp.sizeoutput) + "\n\t\t- maxgrowth parameter = " + str(inp.maxgrowth) + "\n\t\t- step of growth = " + str(inp.stepgrowth))
print("\t\t- type of growth = " + inp.ty + "\n\t\t- n. generated NNs: " + str(inp.ntest) + "\n\tcomputing...")

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
steps = int(inp.maxgrowth/inp.stepgrowth)
outputarray = np.zeros((2*steps, inp.ntest))

# 0.2.3
for gr in range(steps):

	# 0.2.3.1
	growth = (gr + 1)*inp.stepgrowth
	print("\t\t" + str(growth) + ":")

	# 0.2.3.2
	c = 0.8
	ca = torch.ones(inp.nh + 1)
	for j in range(inp.nh + 1): 
		ca[j] = c/hmu(j - 1, growth, inp.ty) 
	cb = 0.2
	sqrtca = torch.sqrt(ca)
	sqrtcb = math.sqrt(cb)

	# 0.2.3.3
	for test in range(inp.ntest):
		model = NeuralNetwork(inp.nh, inp.sizeinput, inp.sizeoutput, growth, inp.ty).to(device)

		# 0.2.3.3.1 
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

		# 0.2.3.3.2
		[outputarray[gr, test], outputarray[gr + 1, test]] = model(sample_input).cpu().detach().numpy()

		# 0.2.3.3.3
		if test%100 == 0:
			print("\t\t\t" + str(test) + "...")
		
	# 0.2.3.4

	# 0.2.3.4.1
	x = outputarray[gr, :]
	y = outputarray[gr + 1, :]
	xspace = 1.2*max(-min(x), max(x))
	yspace = 1.2*max(-min(y), max(y))
	hist, xedges, yedges = np.histogram2d(x, y, bins = inp.nbins, range = [[-xspace, xspace], [-yspace, yspace]], density = True)
	xedgesb = np.insert(xedges.copy(), [0], 0)
	xedgese = np.insert(xedges.copy(), [inp.nbins + 1], 0)
	xmid = ((xedgesb + xedgese)/2)[1:-1]
	yedgesb = np.insert(yedges.copy(), [0], 0)
	yedgese = np.insert(yedges.copy(), [inp.nbins + 1], 0)
	ymid = ((yedgesb + yedgese)/2)[1:-1]

	# 0.2.3.4.1.1
	print("\t\t\t" + str(test + 1) + "!")
	fig = plt.figure(growth)
	ax = fig.add_subplot(projection = "3d")
	xpos, ypos = np.meshgrid(xmid, ymid, indexing = "ij")
	xpos = xpos.ravel()
	ypos = ypos.ravel()
	zpos = np.zeros_like(xpos)
	distx = abs(xmid[1] - xmid[0])
	disty = abs(ymid[1] - ymid[0])
	dx = distx * np.ones_like(zpos)
	dy = disty * np.ones_like(zpos)
	dz = hist.ravel()
	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color = "gray")
	if inp.tex == 1:
		plt.title("3D histogram", fontsize = 10)
		plt.xlabel("$x$")
		plt.ylabel("$y$")
		ax.set_zlabel("$p_{\\mathrm{NN}}(x,y)$")
		plt.savefig(path + "exp_1_3d_hist_" + str(growth) + ".pgf")
	else:
		plt.title("3D histogram", fontsize = 10)
		plt.xlabel("$x$")
		plt.ylabel("$y$")
		ax.set_zlabel("p_NN(x,y)")

	# 0.2.3.4.1.2
	fig = plt.figure(growth + 1)
	ax = fig.add_subplot(projection = "3d")
	cmap_1 = plt.get_cmap("gray")
	xcoord = xpos
	ycoord = ypos
	zcoord = dz
	ax.plot_trisurf(xcoord, ycoord, zcoord, edgecolor = "gray", linewidth = 0.2, antialiased = True, cmap = cmap_1)
	if inp.tex == 1:
		plt.title("tri-surface plot", fontsize = 10)
		plt.xlabel("$x$")
		plt.ylabel("$y$")
		ax.set_zlabel("$p_{\\mathrm{NN}}(x,y)$")
		plt.savefig(path + "exp_1_3d_tsp_" + str(growth) + ".pgf")
	else:
		plt.title("tri-surface plot", fontsize = 10)
		plt.xlabel("x")
		plt.ylabel("y")
		ax.set_zlabel("p_NN(x,y)")

if inp.tex == 0:
	plt.show()

toc()