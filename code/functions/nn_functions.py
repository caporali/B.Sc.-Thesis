# -1 presets
#	-1.1 imported packages
#
# 0 list of NN functions
# 	0.1 hmu
#	0.2 activation
# 	0.3 plot_activation
#		0.3.1 parametri e costruzione dei vettori
#		0.3.2 plot
#	0.4 unison_shuffle
#


# -1

# -1.1
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# 0

# 0.1
def hmu(mu, n, ty):
	# hmu() := funzione di crescita dei layer della NN
	#
	#	input:
	#		mu := numero del layer
	#		n := parametro di crescita
	#		ty := tipo di crescita
	#					"exp" := crescita esponenziale
	#					"const" := crescita costante
	#
	#	output:
	#		hmu(mu, n, ty) := altezza del layer mu nella NN
	#
	#	nota: il passo base per mu == 0 va impostato per dare coerenza ai C_a^mu
	#
	if mu == 0:
		return(1)
	else:
		if ty == "exp":
			return(n ** mu)
		elif ty == "const":
			return(n)

# 0.2
def activation(x, chosen_activation):
	# activation() := funzione che permette di scegliere quale funzione di attivazione applicare ad x
	#
	#	input:
	#		x := parametro a cui applicare la funzione di attivazione
	#		chosen_activation := denotazione assegnata a ciascuna funzione di attivazione
	#
	#	output:
	#		activation(x, chosen_activation) := funzione di attivazione associata a chosen_activation applicata ad x
	#
	if chosen_activation == "ReLU":
		#return torch.clamp(x, min = 0)
		act = nn.ReLU()
		return act(x)
	elif chosen_activation == "ReLU1":
		return torch.clamp(x, min = 0, max = 1)
	elif chosen_activation == "ReLU6":
		#return torch.clamp(x, min = 0, max = 6)
		act = nn.ReLU6()
		return act(x)
	elif chosen_activation == "PRLU3":
		return torch.clamp(x**3, min = 0)
	elif chosen_activation == "Indicator0infty":
		return torch.clamp(torch.sign(x), min = 0)
	elif chosen_activation == "Sigmoid":
		#return 1/(1 + torch.exp(-x))
		act = nn.Sigmoid()
		return act(x)
	elif chosen_activation == "Tanh":
		#return (1/math.pi * (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))) + 1/2
		act = nn.Tanh()
		return act(x)
	else:
		print("error: there is not a " + chosen_activation + " activation function.")

# 0.3 
def plot_activation(chosen_activation, tex, path):
	# plot_activation() := costruisce il plot di una funzione di attivazione
	#
	#	input:
	#		chosen_activation := funzione di attivazione di cui costruire il plot
	#		tex := parametro che controlla l'output di plot_activation()
	#
	#	output:
	#		se tex = 0 (diverso da 1 in realtà) si ha stampa a schermo del plot, altrimenti viene salvato un file pgf
	#
	# 0.3.1
	lim = 5
	num = 100
	x = np.linspace(-lim, lim, num)
	x =  torch.tensor(x)
	ax = activation(x, chosen_activation)

	# 0.3.2
	plt.figure(-1)
	plt.plot(x, ax, "k-")
	plt.xlim(-lim, lim)
	plt.grid()
	plt.gca().set_aspect(aspect = "auto", adjustable = "datalim") # plt.gca().set_aspect(aspect = "equal", adjustable = "datalim")
	if tex == 1:
		matplotlib.use("pgf")
		plt.rcParams.update({
			"pgf.texsystem": "pdflatex",
		    "font.family": "serif",
		    "text.usetex": True,
		    "pgf.rcfonts": False,
			"font.size": 8,
			"axes.axisbelow": True
		})
		plt.xlabel("$x$")
		plt.ylabel( chosen_activation + "$(x)$")
		plt.savefig(path + "activation_function_" + chosen_activation + ".pgf")
	else:
		plt.rcParams.update({ 
		    "font.sans-serif": "Courier New",
			"font.size": 8,
			"axes.axisbelow": True
		})
		plt.xlabel("x")
		plt.ylabel(chosen_activation + "(x)")

# 0.4
def unison_shuffle(a, b):
	# unison_shuffle() := restituisce due copie mescolate all'unisono di a e b, numpy array
	#
	#	input:
	#		a, b := due numpy aray da mescolare
	#
	#	output:
	#		unison_shuffle(a, b) := coppia di array mescolati all'unisono
	#
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]