import numpy as np
import oneflow as flow
from scipy.sparse import coo_matrix
import torch



torch.backends.cuda.matmul.allow_tf32 = True

device = "cuda:0"
# load the sparse matrix and the dense matrix called support
cooRowInd = np.load('./npydata/cooRowInd.npy')
cooColInd = np.load('./npydata/cooColInd.npy')
cooValues = np.load('./npydata/cooValues.npy')
dimension = np.load('./npydata/dimension.npy')
support = np.load('./npydata/support.npy')
rows = dimension[0]
cols = dimension[1]
#!scipy
index = np.argsort(cooRowInd)
cooRowInd = cooRowInd[index]
cooColInd = cooColInd[index]
numpy_y = (coo_matrix((cooValues, (cooRowInd, cooColInd)), shape=(rows, cols))* support)
#print("numpy_coo_mm\n", support)

#!oneflow_coo_mm
device = flow.device("cuda:0")
print("Using {} device".format(device))
flow.set_printoptions(precision=8)
acr = flow.from_numpy(cooRowInd).to(device)
acc = flow.from_numpy(cooColInd).to(device)
acv = flow.from_numpy(cooValues).to(device)
spt = flow.from_numpy(support).to(device)
flow_y = flow._C.spmm_coo(acr, acc, acv, rows, cols, spt)

#print("oneflow_coo_mm\n", spt)

#! pytorch sparse 
torch.set_printoptions(8)
torch_a = torch.sparse_coo_tensor([cooRowInd, cooColInd], cooValues, (rows, cols));
torch_y = torch.sparse.mm(torch_a, torch.from_numpy(support))
#print("torch_coo_mm\n", torch.from_numpy(support)) 

print("compare results between flow_y vs numpy_y")
if(not np.allclose(flow_y.numpy(),numpy_y, 1e-05, 1e-05)):
    print("mismatch!")
    print("flow_y.numpy()\n", flow_y.numpy())
    print("numpy_y\n", numpy_y) 
else:
    print("all elements are equal within 1e-5")
print("------------------------------------------")


print("compare results between torch_y vs numpy_y")
if(not np.allclose(torch_y.numpy(),numpy_y, 1e-05, 1e-05)):
    print("mismatch!")
    print("torch_y.numpy()\n", torch_y.numpy())
    print("numpy_y\n", numpy_y) 
else:
    print("all elements are equal within 1e-5")

print("------------------------------------------")

print("compare results between torch_y vs flow_y")
if(not np.allclose(torch_y.numpy(),flow_y.numpy(), 1e-05, 1e-05)):
    print("mismatch!")
    print("torch_y.numpy()\n", torch_y.numpy())
    print("flow_y.numpy()\n", flow_y.numpy()) 
else:
    print("all elements are equal within 1e-5")
print("------------------------------------------")