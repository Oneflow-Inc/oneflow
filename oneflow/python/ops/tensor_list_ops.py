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
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
from typing import Optional, Sequence, Tuple
import oneflow
import oneflow_api


@oneflow_export("tensor_list_to_tensor_buffer")
def tensor_list_to_tensor_buffer(
    input: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    """This operator converts `TensorList` to `TensorBuffer`. 

    Refer to `Concept Explanation <https://docs.oneflow.org/basics_topics/concept_explanation.html#3tensorbuffer-tensorlist>`_ 
    for more about TensorList. 

    Args:
        input (oneflow_api.BlobDesc): The input `TensorList`. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.mirrored_view())
        @flow.global_function(function_config=func_config)
        def tensorList_to_tensorBuffer_Job(x: tp.ListListNumpy.Placeholder(shape=(2, 5, 4), dtype=flow.float32),
        ) -> tp.ListListNumpy:
            x = flow.tensor_list_to_tensor_buffer(input=x)
            return flow.tensor_buffer_to_tensor_list(x, 
                                                    shape=(5, 4), 
                                                    dtype=flow.float32)

        x = np.random.rand(1, 3, 2).astype(np.float32)
        y = np.random.rand(1, 2, 2).astype(np.float32)
        out = tensorList_to_tensorBuffer_Job([[x, y]])

        # out[0][0].shape (1, 3, 2)

    """
    if name is None:
        name = id_util.UniqueStr("TensorListToBuffer_")

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.tensor_list_to_tensor_buffer_conf, "in", input.unique_name)
    setattr(op_conf.tensor_list_to_tensor_buffer_conf, "out", "out")
    interpret_util.Forward(op_conf)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("tensor_buffer_to_tensor_list")
def tensor_buffer_to_tensor_list(
    input: oneflow_api.BlobDesc,
    shape: Sequence[int],
    dtype: oneflow.dtype,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator converts `TensorBuffer` to `TensorList`. 

    Refer to `Concept Explanation <https://docs.oneflow.org/basics_topics/concept_explanation.html#3tensorbuffer-tensorlist>`_ 
    for more about TensorList. 

    Args:
        input (oneflow_api.BlobDesc): The input Tensor Buffer. 
        shape (Sequence[int]): The shape of input Tensor Buffer. 
        dtype (oneflow.dtype): The data type. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. 
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def tensorBuffer_to_tensorList_Job(x: tp.Numpy.Placeholder(shape=(4, 16, 64, 64), dtype=flow.float32),
        ) -> tp.ListListNumpy:
            x = flow.tensor_to_tensor_buffer(x, 
                                            instance_dims=3)
            out = flow.tensor_buffer_to_tensor_list(input=x, 
                                                    shape=(16, 64, 64), 
                                                    dtype=flow.float32)
            return out

        x = np.random.randn(4, 16, 64, 64).astype(np.float32)
        out = tensorBuffer_to_tensorList_Job(x)

        # out[0][0].shape (1, 16, 64, 64)

    """
    if name is None:
        name = id_util.UniqueStr("TensorBufferToList_")

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.tensor_buffer_to_tensor_list_conf, "in", input.unique_name)
    setattr(op_conf.tensor_buffer_to_tensor_list_conf, "out", "out")
    op_conf.tensor_buffer_to_tensor_list_conf.shape.dim[:] = list(shape)
    setattr(
        op_conf.tensor_buffer_to_tensor_list_conf,
        "data_type",
        oneflow_api.deprecated.GetProtoDtype4OfDtype(dtype),
    )
    interpret_util.Forward(op_conf)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("tensor_list_split")
def tensor_list_split(
    input_tensor_list: oneflow_api.BlobDesc, name: Optional[str] = None
) -> Tuple[oneflow_api.BlobDesc]:
    """This operator splits the input `TensorList`. 

    Args:
        input_tensor_list (oneflow_api.BlobDesc): The input `TensorList`. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        Tuple[oneflow_api.BlobDesc]: A Tuple of `ListNumpy`. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp
        from typing import Tuple


        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.mirrored_view())
        @flow.global_function(function_config=func_config)
        def tensorList_split_Job(x: tp.ListListNumpy.Placeholder(shape=(2, 5, 4), dtype=flow.float32),
        ) -> Tuple[tp.ListNumpy, tp.ListNumpy]:
            return flow.tensor_list_split(x)


        x = np.random.rand(1, 3, 2).astype(np.float32)
        y = np.random.rand(1, 2, 2).astype(np.float32)
        out = tensorList_split_Job([[x, y]])

        # out[0][0].shape (3, 2)
        # out[1][0].shape (2, 2)

    """
    if name is None:
        name = id_util.UniqueStr("TensorListSplit_")

    output_size = input_tensor_list.shape[0]
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.tensor_list_split_conf, "in", input_tensor_list.unique_name)
    op_conf.tensor_list_split_conf.out.extend(
        ["out_{}".format(i) for i in range(output_size)]
    )
    interpret_util.Forward(op_conf)
    ret = []
    for i in range(output_size):
        out_lbi = logical_blob_id_util.LogicalBlobId()
        setattr(out_lbi, "op_name", op_conf.name)
        setattr(out_lbi, "blob_name", "out_{}".format(i))
        ret.append(remote_blob_util.RemoteBlob(out_lbi))
    return tuple(ret)
