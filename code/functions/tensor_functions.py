# -1 presets
#	-1.1 imported packages
#	-1.2 set graphic card 
#
# 0 list of tensor_functions
#	0.1 round_tensor
#	0.2 cast_tensor
#	0.3 print_tensor
#


# -1

# -1.1
import numpy as np
import torch
import torch.nn as nn

# -1.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 0

# 0.1
def round_tensor(tensor, n_digits):
	# round_tensor() := arrotondamento di tensore con n cifre decimali (component-wise)
	#
	#	input:
	#		tensor := tensore
	#		n_digits := pnumero di cifre a cui arrotondare
	#
	#	output:
	#		round_tensor(tensor, n_digits) := tensore arrotondato
	#
	return (tensor * 10**n_digits).round()/(10**n_digits)

# 0.2
def cast_tensor(n):
	# cast_tensor() := cast da numero python a tensore 1-dimensionale di lunghezza 1
	#
	#	input:
	#		n := qualunque numero di python
	#
	#	output:
	#		cast_tensor(n) := tensore 1-dimensionale di lunghezza 1
	#
	return torch.tensor([n]).float().to(device)

# 0.3
def print_tensor(tensor):
	# print_tensor() := pretty print of a torch tensor
	#
	#	input:
	#		tensor := the tensor to print
	#
	#	output:
	#		print_tensor(tensor) := the pretty printable tensor
	#
	return np.around(tensor.cpu().detach().numpy(), decimals = 2)