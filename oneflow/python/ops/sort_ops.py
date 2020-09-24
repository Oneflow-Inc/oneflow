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

from typing import Optional

import oneflow as flow
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("sort")
def sort(
    input: remote_blob_util.BlobDef,
    direction: str = "ASCENDING",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    """This operator sorts the input Blob. 

    Args:
        input (remote_blob_util.BlobDef): A Blob
        direction (str, optional): The direction in which to sort the Blob values. If the direction is "ASCENDING", The order of input will be sorted as ascending, else, the order of input will be sorted as descending. Defaults to "ASCENDING".
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        remote_blob_util.BlobDef: The sorted Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def sort_Job(x: tp.Numpy.Placeholder((5, ))
        ) -> tp.Numpy:
            return flow.sort(input=x, 
                            direction='ASCENDING')

        x = np.array([10, 2, 9, 3, 7]).astype("float32")
        out = sort_Job(x)

        # out [ 2.  3.  7.  9. 10.]

    """
    assert direction in ["ASCENDING", "DESCENDING"]
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Sort_"))
        .Op("sort")
        .Input("in", [input])
        .Output("out")
        .Attr("direction", direction)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("argsort")
def argsort(
    input: remote_blob_util.BlobDef,
    direction: str = "ASCENDING",
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    """This operator sorts the input Blob and return the indices of sorted Blob. 

    Args:
        input (remote_blob_util.BlobDef): A Blob
        direction (str, optional): The direction in which to sort the Blob values. If the direction is "ASCENDING", The order of input will be sorted as ascending, else, the order of input will be sorted as descending. Defaults to "ASCENDING".
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        remote_blob_util.BlobDef: The indices of sorted Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def argsort_Job(x: tp.Numpy.Placeholder((5, ))
        ) -> tp.Numpy:
            return flow.argsort(input=x, 
                                direction='ASCENDING')

        x = np.array([10, 2, 9, 3, 7]).astype("float32")
        out = argsort_Job(x)

        # out [1 3 4 2 0]

    """
    assert direction in ["ASCENDING", "DESCENDING"]
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("ArgSort_")
        )
        .Op("arg_sort")
        .Input("in", [input])
        .Output("out")
        .Attr("direction", direction)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
