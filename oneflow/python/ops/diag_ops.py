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

import oneflow as flow
import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.remote_blob as remote_blob_util
from typing import Optional
import oneflow_api


@oneflow_export("diag")
def diag(
    input: oneflow_api.BlobDesc,
    diagonal: Optional[int] = 0,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
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

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def Diag_Job(input: tp.Numpy.Placeholder((2, 2), dtype=flow.float32),) -> tp.Numpy:
            return flow.diag(input)


        input = np.array([[1.0, 2.0], [3.0, 4.0],], dtype=np.float32)
        out = Diag_Job(input)
        # out [1. 4.]

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
