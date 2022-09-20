import unittest
import numpy as np
import oneflow as flow
from scipy.sparse import coo_matrix


a_coo_row = np.array([0, 0, 0, 1, 2, 2, 2, 3, 3], dtype=np.int32)
a_coo_col = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
a_coo_val = np.array(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32
)
a_rows = 4
a_cols = 4
b = np.array(
    [[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0], [4.0, 8.0, 12.0]],
    dtype=np.float32,
)

device = flow.device("cuda:1")
print("Using {} device".format(device))

acr = flow.from_numpy(a_coo_row).to(device)
acc = flow.from_numpy(a_coo_col).to(device)
acv = flow.from_numpy(a_coo_val).to(device)
bb = flow.from_numpy(b).to(device)
adj = [acr, acc, acv, a_rows, a_cols]
print(adj[0])
flow_y = flow._C.spmm_coo(adj[0], adj[1], adj[2], adj[3], adj[4], bb)
print(flow_y)

numpy_y = (
    coo_matrix((a_coo_val, (a_coo_row, a_coo_col)), shape=(a_rows, a_cols))
    * b
)

print(numpy_y)

