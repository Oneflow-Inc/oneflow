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

from oneflow_api import TensorTuple
from ..framework.tensor import Tensor
from oneflow.python.oneflow_export import oneflow_export
from oneflow_api.autograd import grad as grad_api
from oneflow_api.autograd import backward as backward_api
from typing import Union, Sequence, Tuple


def _convert2tensor_tuple(args: Union[Tensor, Sequence[Tensor], None]):
    if args is None:
        return TensorTuple()
    elif isinstance(args, Tensor):
        tensor_tuple = TensorTuple()
        tensor_tuple.append(args._local_or_consistent_tensor)
        return tensor_tuple
    else:
        return TensorTuple([x._local_or_consistent_tensor for x in args])


@oneflow_export("autograd.grad")
def grad(
    outputs: Union[Tensor, Sequence[Tensor]],
    inputs: Union[Tensor, Sequence[Tensor]],
    out_grads: Union[Tensor, Sequence[Tensor], None],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Tuple[Tensor]:
    in_grads = grad_api(
        _convert2tensor_tuple(outputs),
        _convert2tensor_tuple(inputs),
        _convert2tensor_tuple(out_grads),
        retain_graph,
        create_graph,
    )
    return tuple([Tensor(x) for x in in_grads])


@oneflow_export("autograd.backward")
def backward(
    outputs: Union[Tensor, Sequence[Tensor]],
    out_grads: Union[Tensor, Sequence[Tensor], None],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> None:
    backward_api(
        _convert2tensor_tuple(outputs),
        _convert2tensor_tuple(out_grads),
        retain_graph,
        create_graph,
    )
