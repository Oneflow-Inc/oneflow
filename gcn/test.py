import unittest
import numpy as np
import oneflow as flow
from scipy.sparse import coo_matrix


a_cooRowInd = np.array([0, 0, 0, 1, 2, 2, 2, 3, 3], dtype=np.int32)
a_cooColInd = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
a_cooValues = np.array(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32
)
a_rows = 4
a_cols = 4
b = np.array(
    [[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0], [4.0, 8.0, 12.0]],
    dtype=np.float32,
)

device = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(device))

acr = flow.from_numpy(a_cooRowInd).to(device)
acc = flow.from_numpy(a_cooColInd).to(device)
acv = flow.from_numpy(a_cooValues).to(device)
bb = flow.from_numpy(b).to(device)
flow_y = flow._C.spmm_coo(acr, acc, acv, a_rows, a_cols, bb)
print(flow_y)

numpy_y = (
    coo_matrix((a_cooValues, (a_cooRowInd, a_cooColInd)), shape=(a_rows, a_cols))
    * b
)

print(numpy_y)

