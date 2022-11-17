import unittest
import numpy as np
import oneflow as flow
#import torch
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

#import torch 
import random 


device = "cuda"
nnz = 9
a_rows = 4
a_cols = 4
a_coo_row = np.array([0, 0, 0, 1, 2, 2, 2, 3, 3], dtype=np.int32)
a_coo_col = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
a_coo_val = np.random.rand(nnz).astype(np.float32)
b = np.random.rand(a_cols, a_rows).astype(np.float32)

# acr = flow.tensor(a_coo_row, dtype=flow.int32, device=flow.device(device))
# acc = flow.tensor(a_coo_col, dtype=flow.int32, device=flow.device(device))
# acv = flow.tensor(a_coo_val, dtype=flow.float32, device=flow.device(device))
# bb = flow.tensor(b, dtype=flow.float32, device=flow.device(device))
#flow_y = flow._C.spmm_coo(acr, acc, acv, a_rows, a_cols, bb)
a_coo = coo_matrix((a_coo_val, (a_coo_row, a_coo_col)), shape=(a_rows, a_cols))
np_y = (a_coo * b)
#print(np_y)
a_csr = a_coo.tocsr()
#print(a_csr.indptr, a_csr.indices)

acr = flow.tensor(a_csr.indptr, dtype=flow.int32, device=flow.device(device))
acc = flow.tensor(a_csr.indices, dtype=flow.int32, device=flow.device(device))
acv = flow.tensor(a_csr.data, dtype=flow.float32, device=flow.device(device))
bb = flow.tensor(b, dtype=flow.float32, device=flow.device(device))
flow_y = flow._C.spmm_csr(acr, acc, acv, a_rows, a_cols, bb)
#print(flow_y)

print("compare results between torch_y vs numpy_y")
if(not np.allclose(flow_y.numpy(),np_y, 1e-05, 1e-05)):
    print("mismatch!")
    print("torch_y.numpy()\n", flow_y.numpy())
    print("numpy_y\n", np_y) 
else:
    print("all elements are equal within 1e-5")