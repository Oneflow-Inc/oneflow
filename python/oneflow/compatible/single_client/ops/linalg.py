"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
from typing import Optional

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework import id_util as id_util
from oneflow.compatible.single_client.framework import interpret_util as interpret_util
from oneflow.compatible.single_client.framework import remote_blob as remote_blob_util
from oneflow.core.operator import op_conf_pb2 as op_conf_util
from oneflow.core.register import logical_blob_id_pb2 as logical_blob_id_util


def matmul(
    a: oneflow._oneflow_internal.BlobDesc,
    b: oneflow._oneflow_internal.BlobDesc,
    transpose_a: bool = False,
    transpose_b: bool = False,
    alpha: float = 1.0,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator applies matrix multiplication to two Blobs.

    Args:
        a (oneflow._oneflow_internal.BlobDesc): A Blob
        b (oneflow._oneflow_internal.BlobDesc): A Blob
        transpose_a (bool, optional): Whether to transpose A Blob. Defaults to False.
        transpose_b (bool, optional): Whether to transpose B Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import numpy as np
        import oneflow.compatible.single_client.typing as tp


        @flow.global_function()
        def matmul_Job(A: tp.Numpy.Placeholder((3, 3)),
                    B: tp.Numpy.Placeholder((3, 3))
        ) -> tp.Numpy:
            return flow.linalg.matmul(A, B)


        A = np.array([[1, 0, 0],
                    [0, 1, 1],
                    [0, 0, 1]]).astype(np.float32)
        B = np.array([[3, 4, 5],
                    [6, 7, 8],
                    [9, 10, 11]]).astype(np.float32)
        out = matmul_Job(A, B)

        # output [[ 3.  4.  5.]
        #         [15. 17. 19.]
        #         [ 9. 10. 11.]]

    """
    if name is None:
        name = id_util.UniqueStr("Matmul_")
    assert len(a.shape) >= 2
    assert len(b.shape) >= 2
    if len(a.shape) == len(b.shape):
        if len(a.shape) == 2:
            op = (
                flow.user_op_builder(name)
                .Op("matmul")
                .Input("a", [a])
                .Input("b", [b])
                .Output("out")
                .Attr("transpose_a", transpose_a)
                .Attr("transpose_b", transpose_b)
                .Attr("alpha", float(alpha))
                .Build()
            )
        else:
            op = (
                flow.user_op_builder(name)
                .Op("batch_matmul")
                .Input("a", [a])
                .Input("b", [b])
                .Output("out")
                .Attr("transpose_a", transpose_a)
                .Attr("transpose_b", transpose_b)
                .Attr("alpha", float(alpha))
                .Build()
            )
    else:
        if len(b.shape) != 2:
            raise ValueError(
                "don't support number of dimensions of a being less than number of dimensions of b"
            )
        if transpose_a:
            raise ValueError("don't support tensor a to be tranpose")
        op = (
            flow.user_op_builder(name)
            .Op("broadcast_matmul")
            .Input("a", [a])
            .Input("b", [b])
            .Output("out")
            .Attr("transpose_a", transpose_a)
            .Attr("transpose_b", transpose_b)
            .Attr("alpha", float(alpha))
            .Build()
        )
    return op.InferAndTryRun().SoleOutputBlob()
