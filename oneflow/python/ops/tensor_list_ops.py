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
import oneflow.python.framework.dtype as dtype_util
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


@oneflow_export("tensor_list_to_tensor_buffer")
def tensor_list_to_tensor_buffer(
    input: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
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
    input: remote_blob_util.BlobDef,
    shape: Sequence[int],
    dtype: dtype_util.dtype,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
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
        dtype.oneflow_proto_dtype,
    )
    interpret_util.Forward(op_conf)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("tensor_list_split")
def tensor_list_split(
    input_tensor_list: remote_blob_util.BlobDef, name: Optional[str] = None
) -> Tuple[remote_blob_util.BlobDef]:
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
