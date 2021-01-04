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
from __future__ import absolute_import

import os

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api
from typing import Optional


@oneflow_export("matmul", "linalg.matmul")
def matmul(
    a: oneflow_api.BlobDesc,
    b: oneflow_api.BlobDesc,
    transpose_a: bool = False,
    transpose_b: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator applies matrix multiplication to two Blobs. 

    Args:
        a (oneflow_api.BlobDesc): A Blob
        b (oneflow_api.BlobDesc): A Blob
        transpose_a (bool, optional): Whether to transpose A Blob. Defaults to False.
        transpose_b (bool, optional): Whether to transpose B Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


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
    assert len(a.shape) == len(b.shape)
    assert len(a.shape) >= 2
    if name is None:
        name = id_util.UniqueStr("Matmul_")
    if len(a.shape) == 2:
        op = (
            flow.user_op_builder(name)
            .Op("matmul")
            .Input("a", [a])
            .Input("b", [b])
            .Output("out")
            .Attr("transpose_a", transpose_a)
            .Attr("transpose_b", transpose_b)
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
            .Build()
        )
    return op.InferAndTryRun().RemoteBlobList()[0]
