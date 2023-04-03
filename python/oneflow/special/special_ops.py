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


import oneflow
from oneflow.framework.tensor import Tensor

# avoid redefine error when add_doc


def erf(x: Tensor):
    return oneflow._C.erf(x)


def erfc(x: Tensor):
    return oneflow._C.erfc(x)


def erfinv(x: Tensor):
    return oneflow._C.erfinv(x)


def exp2(x: Tensor):
    return oneflow._C.exp2(x)


def expm1(x: Tensor):
    return oneflow._C.expm1(x)


def log1p(x: Tensor):
    return oneflow._C.log1p(x)


def log_softmax(x: Tensor, dim: int):
    return oneflow._C.log_softmax(x, dim)


def logsumexp(x: Tensor, dim: int, keepdim=False):
    return oneflow._C.logsumexp(x, dim, keepdim)


def round(x: Tensor):
    return oneflow._C.round(x)


def softmax(x: Tensor, dim: int):
    return oneflow._C.softmax(x, dim)
