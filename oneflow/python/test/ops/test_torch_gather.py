import oneflow as flow
import numpy as np
import oneflow.typing as tp
import oneflow.python.framework.id_util as id_util

def torch_gather_test(
    input,
    dim,
    index,
    outtensor = None,
    sparse_grad = False,
    name = None
):
    if name is None:
        name = id_util.UniqueStr("TorchGatherNd_")
    op = (
        flow.user_op_builder(name)
        .Op("torch_gather")
        .Input("in", [input])
        .Input("index", [index])
        .Attr("axis", dim)
        .Attr("sparse_grad", False)
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]

@flow.global_function()
def gather_nd_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.int32), 
                index: tp.Numpy.Placeholder(shape=(2, 2, 2), dtype=flow.int32)
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        gather_nd_blob = torch_gather_test(input=x, 
                                        index=index,
                                        dim=1)
    return gather_nd_blob

x = np.array([[1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]]).astype(np.int32)
indice = np.array([[0, 0], [1, 0], [0, 2]]).astype(np.int32)
out = gather_nd_Job(x, indice)

print(x)
print(out)
