import oneflow as flow
import numpy as np
import oneflow.typing as tp
import oneflow.python.framework.id_util as id_util

@flow.global_function()
def gather_nd_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.int32), 
                index: tp.Numpy.Placeholder(shape=(3,2), dtype=flow.int64)
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        gather_nd_blob = flow.torch_gather(input=x, 
                                        index=index,
                                        dim=1)
    return gather_nd_blob

x = np.array([[1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]]).astype(np.int32)
indice = np.array([[0, 0], [1, 0], [0, 2]]).astype(np.int64)
out = gather_nd_Job(x, indice)

print(x)
print(out)
