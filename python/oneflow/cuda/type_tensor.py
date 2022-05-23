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
import oneflow as flow


__all__ = [
    "HalfTensor",
    "FloatTensor",
    "DoubleTensor",
    "BoolTensor",
    "ByteTensor",
    "CharTensor",
    "IntTensor",
    "LongTensor",
    # TODO: Add support for BFloat16Tensor
]


def HalfTensor(*args, **kwargs):
    r"""
    Creates a Tensor with the dtype of float16 and it has the same parameters as :func:`oneflow.Tensor`.
    """
    return flow.Tensor(*args, **kwargs).to(flow.float16).to("cuda")


def FloatTensor(*args, **kwargs):
    r"""
    Creates a Tensor with the dtype of float32 and it has the same parameters as :func:`oneflow.Tensor`.
    """
    return flow.Tensor(*args, **kwargs).to(flow.float32).to("cuda")


def DoubleTensor(*args, **kwargs):
    r"""
    Creates a Tensor with the dtype of float64 and it has the same parameters as :func:`oneflow.Tensor`.
    """
    return flow.Tensor(*args, **kwargs).to(flow.float64).to("cuda")


def BoolTensor(*args, **kwargs):
    r"""
    Creates a Tensor with the dtype of bool and it has the same parameters as :func:`oneflow.Tensor`.
    """
    return flow.Tensor(*args, **kwargs).to(flow.bool).to("cuda")


def ByteTensor(*args, **kwargs):
    r"""
    Creates a Tensor with the dtype of uint8 and it has the same parameters as :func:`oneflow.Tensor`.
    """
    return flow.Tensor(*args, **kwargs).to(flow.uint8).to("cuda")


def CharTensor(*args, **kwargs):
    r"""
    Creates a Tensor with the dtype of int8 and it has the same parameters as :func:`oneflow.Tensor`.
    """
    return flow.Tensor(*args, **kwargs).to(flow.int8).to("cuda")


def IntTensor(*args, **kwargs):
    r"""
    Creates a Tensor with the dtype of int32 and it has the same parameters as :func:`oneflow.Tensor`.
    """
    return flow.Tensor(*args, **kwargs).to(flow.int32).to("cuda")


def LongTensor(*args, **kwargs):
    r"""
    Creates a Tensor with the dtype of int64 and it has the same parameters as :func:`oneflow.Tensor`.
    """
    return flow.Tensor(*args, **kwargs).to(flow.int64).to("cuda")
