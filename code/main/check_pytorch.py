# -1 presets
#   -1.1 imported packages
#
# 0 inizialization checks
#   0.1 python version
#   0.2 cuda
#
# 1 operational checks
#   1.1 python  
#   1.2 torch
#


# 1

# -1.1
import sys
import random
import torch


# 0 

# 0.1
print("0. inizialization checks \n\n0.1. python version:", sys.version) 
print()

# 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("0.2. using device " + str(device) + ":")
if device.type == "cuda":
    print("	-", torch.cuda.get_device_name(0))
    print("	- memory usage:")
    print("		allocated:", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
    print("		cached:	  ", round(torch.cuda.memory_reserved(0)/1024**3,1), "GB")


# 1

# 1.1
random.seed()
x = random.random()
print("\n1. operational checks\n\n1.1. random number:", x)
print()

# 1.2 
x = torch.rand(5, 3)
print("1.2. random tensor:\n   ", x)