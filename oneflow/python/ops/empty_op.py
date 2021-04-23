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
from typing import Optional, Sequence, Union

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow._oneflow_internal


@oneflow_export("empty")
def constant(
    dtype: Optional[flow.dtype] = None,
    shape: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator creates a uninitialized Blob with specified shape.

    Args:
        dtype (Optional[flow.dtype], optional): The data type of Blob. Defaults to None.
        shape (Optional[Sequence[int]], optional): The shape of Blob. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        NotImplementedError: The data type of value should be int or float.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def empty_Job() -> tp.Numpy:
            empty_blob = flow.empty(value=1.5,
                                        shape=(1, 3, 3),
                                        dtype=flow.float)
            return empty_blob


        out = empty_Job()

        # out [[[1.5 1.5 1.5]
        #       [1.5 1.5 1.5]
        #       [1.5 1.5 1.5]]]

    """
    if name is None:
        name = id_util.UniqueStr("Empty_")

    assert dtype is not None

    if not isinstance(value, (int, float)):
        raise NotImplementedError

    if shape is not None:
        assert isinstance(shape, (list, tuple))
    else:
        shape = []
    return (
        flow.user_op_builder(name)
        .Op("empty")
        .Output("out")
        .Attr("dtype", dtype)
        .Attr("shape", shape)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
