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

import collections
from functools import reduce

import numpy as np
import oneflow as flow
import oneflow_api
from google.protobuf import text_format
from oneflow.python.framework.dtype import convert_proto_dtype_to_oneflow_dtype
from oneflow.python.lib.core.box import Box


class OfBlob(object):
    def __init__(self, of_blob_ptr):
        self.of_blob_ptr_ = of_blob_ptr

    @property
    def dtype(self):
        return convert_proto_dtype_to_oneflow_dtype(
            oneflow_api.Ofblob_GetDataType(self.of_blob_ptr_)
        )

    @property
    def static_shape(self):
        num_axes = oneflow_api.OfBlob_NumAxes(self.of_blob_ptr_)
        dst_ndarray = np.ndarray(num_axes, dtype=np.int64)
        oneflow_api.OfBlob_CopyStaticShapeTo(self.of_blob_ptr_, dst_ndarray)
        return tuple(dst_ndarray.tolist())

    @property
    def shape(self):
        num_axes = oneflow_api.OfBlob_NumAxes(self.of_blob_ptr_)
        dst_ndarray = np.zeros(num_axes, dtype=np.int64)
        oneflow_api.OfBlob_CopyShapeTo(self.of_blob_ptr_, dst_ndarray)
        return tuple(dst_ndarray.tolist())

    def set_shape(self, shape):
        assert isinstance(shape, (list, tuple))
        assert len(shape) == oneflow_api.OfBlob_NumAxes(self.of_blob_ptr_)
        oneflow_api.OfBlob_CopyShapeFrom(
            self.of_blob_ptr_, np.array(shape, dtype=np.int64)
        )

    @property
    def num_axes(self):
        return oneflow_api.OfBlob_NumAxes(self.of_blob_ptr_)

    @property
    def is_dynamic(self):
        return oneflow_api.OfBlob_IsDynamic(self.of_blob_ptr_)

    def CopyToNdarray(self):
        return self._CopyToNdarray()

    def CopyFromNdarray(self, src_ndarray):
        if self.is_dynamic:
            self.set_shape(src_ndarray.shape)
        else:
            shape_tensor = np.zeros(self.num_axes, dtype=np.int64)
            oneflow_api.OfBlob_CopyShapeTo(self.of_blob_ptr_, shape_tensor)
            shape = tuple(shape_tensor.tolist())
            print("HAHA cclog: numpy shape = ", src_ndarray.shape)
            print("HAHA cclog: ofblob shape = ", shape)
            assert src_ndarray.shape == shape
        return self._CopyBodyFromNdarray(src_ndarray)

    def _CopyBodyFromNdarray(self, src_ndarray):
        assert not self.is_dynamic
        method_name = oneflow_api.Dtype_GetOfBlobCopyFromBufferFuncName(
            oneflow_api.deprecated.GetProtoDtype4OfDtype(self.dtype)
        )
        copy_method = getattr(oneflow_api, method_name)
        copy_method(self.of_blob_ptr_, src_ndarray)

    def _CopyToNdarray(self):
        method_name = oneflow_api.Dtype_GetOfBlobCopyToBufferFuncName(
            oneflow_api.deprecated.GetProtoDtype4OfDtype(self.dtype)
        )
        copy_method = getattr(oneflow_api, method_name)
        shape_tensor = np.zeros(self.num_axes, dtype=np.int64)
        oneflow_api.OfBlob_CopyShapeTo(self.of_blob_ptr_, shape_tensor)
        shape = tuple(shape_tensor.tolist())
        print("HAHA cclog: CopyToNdarray ofblob shape = ", shape)
        tensor = np.zeros(
            shape, dtype=flow.convert_oneflow_dtype_to_numpy_dtype(self.dtype)
        )
        copy_method(self.of_blob_ptr_, tensor)
        print("HAHA cclog: CopyToNdarray tensor shape = ", tensor.shape)
        print("HAHA cclog: CopyToNdarray tensor id", id(tensor))
        return [tensor]
