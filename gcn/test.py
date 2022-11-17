import unittest
import numpy as np
import oneflow as flow
#import torch
from scipy.sparse import coo_matrix, csr_matrix
import torch 
import random 

torch.backends.cuda.matmul.allow_tf32 = True

    
# rows = 4
# cols = 6
# nnz = rows + cols
# cooRowInd = np.zeros(nnz, dtype=np.int32)
# cooColInd = np.zeros(nnz, dtype=np.int32) 
# cooValues = np.zeros(nnz, dtype=np.float32) 

# for n in range(nnz):
#     cooRowInd[n] = random.randint(0,rows-1)
#     cooColInd[n] = random.randint(0,cols-1)
#     cooValues[n] = random.random()
# cooRowInd.sort()
# cooColInd.sort()
# b = np.random.rand(cols, rows).astype(np.float32)
# # print(cooRowInd)
# # print(cooColInd)
# # print(cooValues)

# #!scipy
# a = coo_matrix((cooValues, (cooRowInd, cooColInd)), shape=(rows, cols))
# numpy_y = (
#     a * b
# )
# #print("numpy_coo_mm\n", numpy_y)

# #!oneflow_coo_mm
# device = flow.device("cuda:0")
# print("Using {} device".format(device))
# flow.set_printoptions(precision=8)
# a=a.tocsr()
# acr = flow.from_numpy(a.indptr).to(device)
# acc = flow.from_numpy(a.indices).to(device)
# acv = flow.from_numpy(a.data).to(device)
# bb = flow.from_numpy(b).to(device)
# adj = [acr, acc, acv, rows, cols]
# flow_y = flow._C.spmm_csr(adj[0], adj[1], adj[2], adj[3], adj[4], bb, False, False)

# #print("oneflow_csr_mm\n", flow_y)

# #! pytorch sparse 
# torch.set_printoptions(8)
# torch_a = torch.sparse_coo_tensor([cooRowInd, cooColInd], cooValues, (rows, cols));
# torch_y = torch.sparse.mm(torch_a, torch.from_numpy(b))
# #print("torch_coo_mm\n", torch_y) 

# print("compare results between flow_y vs numpy_y")
# if(not np.allclose(flow_y.numpy(),numpy_y, 1e-05, 1e-05)):
#     print("mismatch!")
#     print("flow_y.numpy()\n", flow_y.numpy())
#     print("numpy_y\n", numpy_y) 
# else:
#     print("all elements are equal within 1e-5")


# print("compare results between torch_y vs numpy_y")
# if(not np.allclose(torch_y.numpy(),numpy_y, 1e-05, 1e-05)):
#     print("mismatch!")
#     print("torch_y.numpy()\n", torch_y.numpy())
#     print("numpy_y\n", numpy_y) 
# else:
#     print("all elements are equal within 1e-5")
indptr = np.array([ 0, 1, 3, 4])
indices = np.array([0, 0, 1, 1])
data = np.array([1, 2, 3, 4])
a=csr_matrix((data, indices, indptr), shape=(3, 2)).toarray()
b = np.array([1,  1,  1, 1, 0, 1, 0, 0, 1])
b = np.reshape(b, (3, 3))
a = a.T
print(a.shape)
print(a, "\n---------------------------")

print(b.shape)
print(b, "\n---------------------------")

y = a @ b
print(y)
