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
import oneflow as flow


def norm(self, ord=None, dim=None, keepdim=False, dtype=None):
    return flow._C.norm(self, ord, dim, keepdim, dtype=dtype)


def vector_norm(self, ord=2, dim=None, keepdim=False, dtype=None):
    return flow._C.vector_norm(self, ord, dim, keepdim, dtype=dtype)


def matrix_norm(self, ord="fro", dim=(-2, -1), keepdim=False, dtype=None):
    return flow._C.matrix_norm(self, ord, dim, keepdim, dtype=dtype)


def diagonal(self, input, offset=0, dim1=-2, dim2=-1):
    """
    Alias for :func:`oneflow.diagonal` with defaults :attr:`dim1`\ `= -2`, :attr:`dim2`\ `= -1`.
    """
    return flow._C.diagonal(self, input, offset=offset, dim1=dim1, dim2=dim2)
