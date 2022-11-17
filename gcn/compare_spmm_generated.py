import unittest
import numpy as np
import oneflow as flow
#import torch
from scipy.sparse import coo_matrix
import torch 
import random 

torch.backends.cuda.matmul.allow_tf32 = True

# randomly generate a sparse matrix in COO and a dense matrix named support
rows = 4
cols = 6
nnz = rows + cols
cooRowInd = np.zeros(nnz, dtype=np.int32)
cooColInd = np.zeros(nnz, dtype=np.int32) 
cooValues = np.zeros(nnz, dtype=np.float32) 

for n in range(nnz):
    cooRowInd[n] = random.randint(0,rows-1)
    cooColInd[n] = random.randint(0,cols-1)
    cooValues[n] = random.random()
index = np.argsort(cooRowInd)
cooRowInd = cooRowInd[index]
cooColInd = cooColInd[index]
print(cooRowInd)
print(cooColInd)

""" start = 0
while start < nnz-1:
    next = start+1
    print(start, next)
    while(cooRowInd[start]==cooRowInd[next]):
        next = next + 1
    np.sort(cooColInd[start: next])
    start = next

print(cooColInd) """

support = np.random.rand(cols, rows).astype(np.float32)
# print(cooRowInd)
# print(cooColInd)
# print(cooValues)

#!scipy
numpy_y = (coo_matrix((cooValues, (cooRowInd, cooColInd)), shape=(rows, cols))* support)
#print("numpy_coo_mm\n", numpy_y)

#!oneflow_coo_mm
device = flow.device("cuda:0")
print("Using {} device".format(device))
flow.set_printoptions(precision=8)

acr = flow.from_numpy(cooRowInd).to(device)
acc = flow.from_numpy(cooColInd).to(device)
acv = flow.from_numpy(cooValues).to(device)
spt = flow.from_numpy(support).to(device)
flow_y = flow._C.spmm_coo(acr, acc, acv, rows, cols, spt)

#print("oneflow_coo_mm\n", flow_y)

#! pytorch sparse 
torch.set_printoptions(8)
torch_a = torch.sparse_coo_tensor([cooRowInd, cooColInd], cooValues, (rows, cols));
torch_y = torch.sparse.mm(torch_a, torch.from_numpy(support))
#print("torch_coo_mm\n", torch_y) 

print("compare results supportetween flow_y vs numpy_y")
if(not np.allclose(flow_y.numpy(),numpy_y, 1e-05, 1e-05)):
    print("mismatch!")
    print("flow_y.numpy()\n", flow_y.numpy())
    print("numpy_y\n", numpy_y) 
else:
    print("all elements are equal within 1e-5")
print("------------------------------------------")


print("compare results supportetween torch_y vs numpy_y")
if(not np.allclose(torch_y.numpy(),numpy_y, 1e-05, 1e-05)):
    print("mismatch!")
    print("torch_y.numpy()\n", torch_y.numpy())
    print("numpy_y\n", numpy_y) 
else:
    print("all elements are equal within 1e-5")

print("------------------------------------------")

print("compare results supportetween torch_y vs flow_y")
if(not np.allclose(torch_y.numpy(),flow_y.numpy(), 1e-05, 1e-05)):
    print("mismatch!")
    print("torch_y.numpy()\n", torch_y.numpy())
    print("flow_y.numpy()\n", flow_y.numpy()) 
else:
    print("all elements are equal within 1e-5")
print("------------------------------------------")
