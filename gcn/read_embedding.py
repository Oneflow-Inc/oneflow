import os
from oneflow.one_embedding import make_persistent_table_reader
import numpy as np
import oneflow as flow


ids = np.random.randint(0, 1000, (100, 3), dtype=np.int64)
ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
print(ids_tensor.size())