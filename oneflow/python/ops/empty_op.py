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
from oneflow._oneflow_internal.distribute import SplitDistribute, BroadcastDistribute
import re


@oneflow_export("empty")
def empty(
    dtype: Optional[flow.dtype] = None,
    shape: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
    distribute: Optional[Union[SplitDistribute, BroadcastDistribute, str]] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator creates a uninitialized Blob with specified shape.

    Args:
        dtype (Optional[flow.dtype], optional): The data type of Blob. Defaults to None.
        shape (Optional[Sequence[int]], optional): The shape of Blob. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.
        distribute: The distribute attribute which can be one of SplitDistribute, BroadcastDistribute or in str format "S(N)", "B".

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        Example 1:

        @flow.global_function()
        def empty_Job() -> tp.Numpy:
            empty_blob = flow.empty(shape=(1, 3, 3),
                                    dtype=flow.float)
            return empty_blob
        out = empty_Job() # out tensor with shape (1, 3, 3) and data uninitialized

        Example 2:

        flow.config.gpu_device_num(2) #set gpu num to 2
        @flow.global_function()
        def empty_Job() -> tp.Numpy:
            empty_blob = flow.empty(shape=(10, 3, 3),
                                    dtype=flow.float,
                                    distribute="S(0)") #split at axis 0
            return empty_blob

        Example 3:

        flow.config.gpu_device_num(2) #set gpu num to 2
        @flow.global_function()
        def empty_Job() -> tp.Numpy:
            empty_blob = flow.empty(shape=(10, 3, 3),
                                    dtype=flow.float,
                                    distribute=flow.distribute.split(0)) # same as "S(0)",
            return empty_blob

    """

    def _distribute_to_str(dist):
        dist_str = ""
        if dist is None:
            pass
        elif type(dist) is oneflow._oneflow_internal.distribute.SplitDistribute:
            dist_str = "S({})".format(dist.axis)
        elif type(dist) is oneflow._oneflow_internal.distribute.BroadcastDistribute:
            dist_str = "B"
        else:
            raise ValueError("unsupported distribute")
        return dist_str

    if name is None:
        name = id_util.UniqueStr("Empty_")

    assert dtype is not None

    if shape is not None:
        assert isinstance(shape, (list, tuple))
    else:
        shape = []
    if distribute is None:
        sbp_parallel = ""
    elif isinstance(distribute, str):
        assert (
            re.match("^S\(\d+\)$", distribute) is not None or distribute == "B"
        ), "The distribute argument can only be 'S(N)'(N is a integer number) or 'B' when its type is str"
        sbp_parallel = distribute
    elif isinstance(distribute, BroadcastDistribute) or isinstance(
        distribute, SplitDistribute
    ):
        sbp_parallel = _distribute_to_str(distribute)
    else:
        raise ValueError("Wrong distribute value")

    return (
        flow.user_op_builder(name)
        .Op("empty")
        .Output("out")
        .Attr("dtype", dtype)
        .Attr("shape", shape)
        .Attr("sbp_parallel", sbp_parallel)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
