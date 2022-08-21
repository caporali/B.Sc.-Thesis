# input from input_exp_2
# 	a scelta della funzione da approssimare
#		a.1 	fstr := stringa contenente la funzione
#		a.2 	f(x) := funzione in python
#		a.3.1 	a := estremo sinistro dell'intervallo di approssimazione
#		a.3.2 	b := estremo destro dell'intervallo di approssimazione
#		a.4 	nsample := numero di campioni da generare dalla funzione per l'approssimazione
# 	b parametri NN
#		b.5 	nh := numero di hidden layer 
#		b.6 	sizeinput := taglia dell'input
#		b.7 	sizeoutput := taglia dell'output
#		b.8 	act := funzione d'attivazione
#		b.9 	growth := parametro da cui dipende l'ampiezza degli hidden layer (parametro di crescita)
#		b.10 	ty := tipo di crescita dell'ampiezza degli hidden layers 
# 	c parametri per il training
#		c.11 	nepoch := numero massimo di volte che viene effettuato il training con lo stesso set di nsample input
#		c.12 	err := errore in norma uniforme tollerabile (se uniform_norm < err c'è uscita prima dell'ultima epoch)
#		c.13 	chosen_lr := learning rate optimizer, parametro per la SDG
# 	d parametri per plot
#		d.14 	loss_track := tiene traccia della loss per comprenderne l'andamento e ottimizzare il learning rate (1 per attivare)
#		d.15 	tex := variabile che determina se i plot saranno visibili o in pgf (rispettivamente 0 e 1)
#		d.16 	act_print := variabile per la stampa della funzione di attivazione (1 per la stampa)
#
# -1 presets
#	-1.1 imported packages
#	-1.2 prevent cache creation
#	-1.3 input
#	-1.4 set path to parent directory
#	-1.5 set graphic card 
#	-1.6 sets matplotlib
#
# 0 costruzione ed inizializzazione NN
# 	0.1 costruzione di f(x) e f(sample) (con sample vettore di nsample campioni di f)
#	0.2 costruzione NN 
#	0.3 assegnazione preliminare parametri NN
#		0.3.1 \hat{C}_A^\mu, C_A^\mu e C_b^\mu
#		0.3.2 assegnazione parametri test-esima NN (in accordo con la teoria 0.3.1)
#
# 1 training
#	1.1 definizione di loss_function e optimizer per training (rispettivamente MSE e SDG) 
#	1.2 stampa parametri
#	1.3 epochs
#		1.3.1 pretty print epoch corrente
#		1.3.2 fase di training
#			1.3.2.1 computazione di input ed exact_output 
#			1.3.2.2 farward propagation (computo di output e loss)
#			1.3.2.3 backward propagation [3 fasi:
#				- clear gradients			model.zero_grad();
#				- compute gradients			loss.backward();
#				- update the parameters		optimizer.step();]
#		1.3.3 loading epoch
#		1.3.4 condizione di uscita anticipata
#		1.3.5 analisi della loss function
#
# 2 tests
# 	2.1 applicazione della NN a x (x \in [a,b])
#	2.2 computo della norma uniforme della differenza tra modelf ed f
#	2.3 save model
#	2.4 plot
#


# -1

# -1.1
import sys
import os
import math
import numpy as np
import random
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

tic()

# 0.1
num = int((inp.b - inp.a)*50)
x = np.linspace(inp.a, inp.b, num)
fx = inp.f(x)
modelfx = np.zeros_like(fx)
sample = inp.a + (inp.b - inp.a)*np.random.rand(inp.nsample) 
f_sample = inp.f(sample) 
modelf_sample = np.zeros_like(f_sample)

# 0.2
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

# 0.3

# 0.3.1
c = 0.8
ca = torch.ones(inp.nh + 1)
for j in range(inp.nh + 1): 
	ca[j] = c/hmu(j - 1, inp.growth, inp.ty) 
cb = 0.2
sqrtca = torch.sqrt(ca)
sqrtcb = math.sqrt(cb)

# 0.3.2
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


# 1

loss_mat = np.zeros((inp.nsample, inp.nepoch))
uniform_norm = 0

# 1.1 
loss_function = nn.MSELoss(reduction = "mean")
optimizer = optim.SGD(model.parameters(), lr = inp.chosen_lr)

# 1.2
print("0. collecting data\n\n0.1. sampling of a continous function: \n\t- f(x) = " + inp.fstr + " \n\t- range = [" + str(inp.a) + ","  + str(inp.b) + "]")
print("\t- n. samples = " + str(inp.nsample) + "\n\n" + "0.2. NN structure: \n\t- n. di hidden layers = " + str(inp.nh) + "\n\t- input size = " + str(inp.sizeinput))
print("\t- output size = " + str(inp.sizeoutput) + "\n\t- activation function = " + inp.act + "\n\t- growth parameter = " + str(inp.growth) + "\n\t- type of growth = " + inp.ty)
print("\n1. training\n\n1.1. training parameters: \n\t- max n. of epoch = " + str(inp.nepoch) + " \n\t- chosen learning rate = " + str("{:.2e}".format(inp.chosen_lr)))
print("\tcomputing...")

# 1.3
c_epoch = 0
for epoch in range(inp.nepoch):

	# 1.3.1
	if c_epoch%5 == 0:
		print("\t\tepoch [" + f"{(epoch + 1):3d}" + "," + f"{(min((epoch + 5), inp.nepoch)):3d}" + "/" + f"{inp.nepoch:3d}" + "]:\t", end = "")
	c_epoch = c_epoch + 1

	shuffle_sample, shuffle_f_sample = unison_shuffle(sample, f_sample)
	for s in range(inp.nsample):

		# 1.3.2

		# 1.3.2.1 
		input = cast_tensor(shuffle_sample[s])
		exact_output = cast_tensor(shuffle_f_sample[s])

		# 1.3.2.2
		output = model(input)
		loss = loss_function(output, exact_output)
		loss_mat[s, epoch] = loss

		# 1.3.2.3
		model.zero_grad()
		loss.backward()
		optimizer.step()

		# 1.3.3
		if s + 1 == inp.nsample and (s + 1)%(inp.nsample/10) == 0 and (c_epoch%5 == 0 or c_epoch == inp.nepoch):
			print("=")
		elif s + 1 == inp.nsample and (s + 1)%(inp.nsample/10) == 0:
			print("= ", end = "")
		elif (s + 1)%(inp.nsample/10) == 0:
			print("=", end = "")

	# 1.3.4
	for var in range(num):
		modelfx[var] = model(cast_tensor(x[var])).item()
	uniform_norm = np.max(np.abs(modelfx - fx)).item()
	if uniform_norm < inp.err:
		print("\n\t\texecution interrupted: \n\t\t\t- current epoch = " + str(epoch + 1) + "\n\t\t\t- uniform_norm < err : " + str(round(uniform_norm, 3)) + " < " + str(inp.err))
		break

# 1.3.5
if inp.loss_track == 1:
	for r in range(inp.nsample):
		loss_mat[0,:] = loss_mat[0,:] + loss_mat[r,:]
	loss_mat[0,:] = loss_mat[0,:]/inp.nsample
	plt.semilogy(np.arange(1, epoch + 2), loss_mat[0, 0:(epoch + 1)], "-k")
	plt.title("current learning rate: lr = " + str("{:.2e}".format(inp.chosen_lr)) + "\n")
	plt.xlabel("epoch")
	plt.ylabel("loss_mat average")
	if inp.tex == 1:
		plt.savefig(path + "exp_2_loss_" + str(inp.switch) + ".pgf")


# 2

# 2.1
for var in range(num):
	modelfx[var] = model(cast_tensor(x[var])).item()
for var in range(inp.nsample):
	modelf_sample[var] = model(cast_tensor(sample[var])).item()

# 2.2
uniform_norm = np.max(np.abs(modelfx - fx)).item()
uniform_norm_sample = np.max(np.abs(modelf_sample - f_sample)).item()
print("\n2. tests\n\t- ||modelf(x) - f(x)||_∞ =", uniform_norm, "\n\t- ||modelf_sample(x) - f_sample(x)||_∞ =", uniform_norm_sample)

toc()

# 2.3
torch.save(model.state_dict(), "model.pt")

# 2.4
print("\n2.1. plot:\n\t- scatter (gray)\n\t- f(x) (green)\n\t- modelf(x) (black)")
if inp.act_print == 1:
	plot_activation(inp.act, inp.tex, path)
plt.figure(0)
plt.plot(x, fx, "g-")
plt.scatter(sample, f_sample, c = "gray", marker = ".")
plt.plot(x, modelfx, "k-")
plt.xlim(inp.a, inp.b)
plt.grid()
if inp.tex == 1:
	plt.title("approximation of $f(x) = " + inp.fstr + "$ with NNs: modelf$(x)$\n", fontsize = 10)
	plt.xlabel("x")
	plt.ylabel("\\texttt{scatter} / $f(x)$ / modelf$(x)$")
	plt.savefig(path + "exp_2_approximation_" + str(inp.switch) + ".pgf")
else:
	plt.title("approximation of f(x) = " + inp.fstr + " with NNs: modelf(x)\n", fontsize = 10)
	plt.xlabel("x")
	plt.ylabel("scatter / f(x) / modelf(x)")
	plt.show()
