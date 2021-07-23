from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework import id_util as id_util
from oneflow.compatible.single_client.python.framework import (
    remote_blob as remote_blob_util,
)
from typing import Optional
import oneflow._oneflow_internal


def diag(
    input: oneflow._oneflow_internal.BlobDesc,
    diagonal: Optional[int] = 0,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator compute diagonal. 

    If input is a vector, then returns a square matrix with the elements of input as the diagonal.
    If input is a matrix, then returns a vector with the diagonal elements of input.
    Args:
        input (remote_blob_util.BlobDef): The input Blob.
        diagonal (Optional[int], 0): The diagonal to consider. If diagonal = 0, it is the main diagonal. If diagonal > 0, it is above the main diagonal. If diagonal < 0, it is below the main diagonal. Defaults to 0.

    Returns:
        remote_blob_util.BlobDef: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow.compatible.single_client as flow
        import numpy as np
        import oneflow.compatible.single_client.typing as tp


        @flow.global_function()
        def Diag_Job(input: tp.Numpy.Placeholder((3, 3), dtype=flow.float32),) -> tp.Numpy:
            return flow.diag(input)


        input = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0],], dtype=np.float32)
        out = Diag_Job(input)
        # out [1. 5. 9.]

    """
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Diag_"))
        .Op("diag")
        .Input("in", [input])
        .Attr("diagonal", int(diagonal))
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
