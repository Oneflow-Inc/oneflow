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
import oneflow.core.serving.tensor_pb2 as tensor_pb
import oneflow.core.common.data_type_pb2 as data_type_pb
import numpy as np

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("serving.array_to_tensor_proto")
def array_to_tensor_proto(ndarray, tensor_proto=None):
    if tensor_proto is None:
        tensor_proto = tensor_pb.TensorProto()

    tensor_proto.shape.dim.extend(ndarray.shape)
    if ndarray.dtype == np.float32:
        tensor_proto.data_type = data_type_pb.kFloat
        tensor_proto.float_list.value.extend(ndarray.flat)
    elif ndarray.dtype == np.float64:
        tensor_proto.data_type = data_type_pb.kDouble
        tensor_proto.double_list.value.extend(ndarray.flat)
    elif ndarray.dtype == np.int32:
        tensor_proto.data_type = data_type_pb.kInt32
        tensor_proto.int32_list.value.extend(ndarray.flat)
    elif ndarray.dtype == np.int64:
        tensor_proto.data_type = data_type_pb.kInt64
        tensor_proto.int64_list.value.extend(ndarray.flat)
    elif ndarray.dtype == np.uint8:
        # TODO: support convert other types to bytes
        raise NotImplementedError
    else:
        raise NotImplementedError
    return tensor_proto


@oneflow_export("serving.tensor_proto_to_array")
def tensor_proto_to_array(tensor_proto):
    if tensor_proto.data_type == data_type_pb.kFloat:
        arr = np.array(tensor_proto.float_list.value).astype(np.float32)
    elif tensor_proto.data_type == data_type_pb.kDouble:
        arr = np.array(tensor_proto.double_list.value).astype(np.float64)
    elif tensor_proto.data_type == data_type_pb.kInt32:
        arr = np.array(tensor_proto.int32_list.value).astype(np.int32)
    elif tensor_proto.data_type == data_type_pb.kInt64:
        arr = np.array(tensor_proto.int64_list.value).astype(np.int64)
    elif tensor_proto.data_type == data_type_pb.kUInt8:
        # TODO: support convert other types from bytes
        raise NotImplementedError
    else:
        raise NotImplementedError
    return arr
