"""
Copyright 2020 The OneFlow Authors. All rights reserved.  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from typing import Sequence

import oneflow
import oneflow_api
import oneflow.python.framework.dtype as dtype_util
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("tensor")
class Tensor(oneflow_api.Tensor):
    def __init__(
        self,
        shape: tuple=(),
        dtype: dtype_util.dtype=dtype_util.float,
        sbp=None,
    ) -> None:
        of_dtype = dtype_util.convert_oneflow_dtype_to_proto_dtype(dtype)
        oneflow_api.Tensor.__init__(self, shape, of_dtype)

    @property
    def dtype(self):
        return dtype_util.convert_proto_dtype_to_oneflow_dtype(self.get_dtype())

