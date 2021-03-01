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
import oneflow_api


@oneflow_export("constant")
def constant(
    value: Union[int, float],
    dtype: Optional[flow.dtype] = None,
    shape: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator creates a constant Blob. 

    Args:
        value (Union[int, float]): The constant value of Blob.
        dtype (Optional[flow.dtype], optional): The data type of Blob. Defaults to None.
        shape (Optional[Sequence[int]], optional): The shape of Blob. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        NotImplementedError: The data type of value should be int or float. 

    Returns:
        oneflow_api.BlobDesc: The result blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def constant_Job() -> tp.Numpy:
            constant_blob = flow.constant(value=1.5, 
                                        shape=(1, 3, 3), 
                                        dtype=flow.float)
            return constant_blob


        out = constant_Job()

        # out [[[1.5 1.5 1.5]
        #       [1.5 1.5 1.5]
        #       [1.5 1.5 1.5]]]

    """
    if name is None:
        name = id_util.UniqueStr("Constant_")
    assert value is not None
    assert dtype is not None

    if not isinstance(value, (int, float)):
        raise NotImplementedError

    if isinstance(value, float):
        is_floating_value = True
        floating_value = float(value)
        integer_value = int(0)
    else:
        is_floating_value = False
        floating_value = float(0)
        integer_value = int(value)
    if shape is not None:
        assert isinstance(shape, (list, tuple))
    else:
        shape = []
    return (
        flow.user_op_builder(name)
        .Op("constant")
        .Output("out")
        .Attr("floating_value", floating_value)
        .Attr("integer_value", integer_value)
        .Attr("is_floating_value", is_floating_value)
        .Attr("dtype", dtype)
        .Attr("shape", shape)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("constant_scalar")
def constant_scalar(
    value: Union[int, float],
    dtype: Optional[flow.dtype] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator creates a constant scalar Blob. 

    Args:
        value (Union[int, float]): The constant value of Blob.
        dtype (Optional[flow.dtype], optional): The data type of Blob. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def constant_scalar_Job() -> tp.Numpy:
            constant_scalar = flow.constant_scalar(value=2.5, 
                                                dtype=flow.float)
            return constant_scalar


        out = constant_scalar_Job()

        # out [2.5]

    """
    return flow.constant(value, dtype=dtype, shape=[1])


@oneflow_export("constant_like")
def constant_like(
    like: oneflow_api.BlobDesc,
    value: Union[int, float],
    dtype: Optional[flow.dtype] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator creates a constant Blob that has the same shape as `like`. 

    Args:
        like (oneflow_api.BlobDesc): A Blob. 
        value (Union[int, float]): The constant value of Blob.
        dtype (Optional[flow.dtype], optional): The data type of Blob. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        NotImplementedError: The data type of value should be int or float. 

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def constant_like_Job() -> tp.Numpy:
            constant_blob = flow.constant(value=1.5, 
                                        shape=(1, 3, 3), 
                                        dtype=flow.float)
            constant_like_blob = flow.constant_like(like=constant_blob, 
                                                    value=5.5, 
                                                    dtype=flow.float)
            return constant_like_blob


        out = constant_like_Job()

        # out [[[5.5 5.5 5.5]
        #       [5.5 5.5 5.5]
        #       [5.5 5.5 5.5]]]

    """
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ConstantLike_"),
    )
    setattr(op_conf.constant_like_conf, "like", like.unique_name)
    if isinstance(value, int):
        op_conf.constant_like_conf.int_operand = value
    elif isinstance(value, float):
        op_conf.constant_like_conf.float_operand = value
    else:
        raise NotImplementedError
    if dtype is not None:
        setattr(
            op_conf.constant_like_conf,
            "data_type",
            oneflow_api.deprecated.GetProtoDtype4OfDtype(dtype),
        )
    setattr(op_conf.constant_like_conf, "out", "out")
    interpret_util.Forward(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("ones_like")
def ones_like(
    like: oneflow_api.BlobDesc,
    dtype: Optional[flow.dtype] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator creates a Blob with all elements set to `1` that has the same shape as `like`.

    Args:
        like (oneflow_api.BlobDesc): A Blob. 
        dtype (Optional[flow.dtype], optional): The data type of Blob. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def ones_like_Job() -> tp.Numpy:
            constant_blob = flow.constant(value=1.5, 
                                        shape=(1, 3, 3), 
                                        dtype=flow.float)
            ones_like_blob = flow.ones_like(like=constant_blob, 
                                            dtype=flow.float)
            return ones_like_blob


        out = ones_like_Job()

        # out [[[1. 1. 1.]
        #       [1. 1. 1.]
        #       [1. 1. 1.]]]

    """
    return constant_like(like, 1, dtype=dtype, name=name)


@oneflow_export("zeros_like")
def zeros_like(
    like: oneflow_api.BlobDesc,
    dtype: Optional[flow.dtype] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator creates a Blob that has the same shape as `like` whose all elements are set to `0`. 

    Args:
        like (oneflow_api.BlobDesc): A Blob. 
        dtype (Optional[flow.dtype], optional): The data type of Blob. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def zeros_like_Job() -> tp.Numpy:
            constant_blob = flow.constant(value=1.5, 
                                        shape=(1, 3, 3), 
                                        dtype=flow.float)
            zeros_like_blob = flow.zeros_like(like=constant_blob, 
                                            dtype=flow.float)
            return zeros_like_blob


        out = zeros_like_Job()

        # out [[[0. 0. 0.]
        #       [0. 0. 0.]
        #       [0. 0. 0.]]]

    """
    return constant_like(like, 0, dtype=dtype, name=name)
