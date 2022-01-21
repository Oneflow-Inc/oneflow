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
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


def in_top_k_op(targets, predictions, k):

    return flow._C.in_top_k(targets, predictions, k=k)


def in_top_k_op_tensor(targets, predictions, k):
    """
    in_top_k() -> Tensor

    See :func:`oneflow.in_top_k`

    """
    return flow._C.in_top_k(targets, predictions, k=k)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
