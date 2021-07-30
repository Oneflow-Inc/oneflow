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


class ToConsistent(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, sbp, placement):
        if isinstance(sbp, flow.sbp.sbp):
            sbp = [sbp]
        else:
            assert isinstance(sbp, (list, tuple))
            sbp = list(sbp)
        return flow.F.to_consistent(x, sbp=sbp, placement=placement)


@register_tensor_op("to_consistent")
def to_consistent_op(input, sbp, placement):
    """Cast a local tensor to consistent tensor or (TODO)cast a
    consistent tensor to another consistent tensor with 
    different sbp or placement


    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> placement = flow.placement("cpu", {0:range(1)})
        >>> output_tensor = input.to_consistent([flow.sbp.split(0)], placement)
        >>> output_tensor.is_consistent
        True
    """
    return ToConsistent()(input, sbp, placement)


class ToLocal(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow.F.to_local(x)


@register_tensor_op("to_local")
def to_local_op(input):
    """Returns the local tensor of a consistent tensor.


    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> placement = flow.placement("cpu", {0:range(1)})
        >>> consistent_tensor = input.to_consistent([flow.sbp.split(0)], placement)
        >>> consistent_tensor.to_local()
        tensor([0.5, 0.6, 0.7], dtype=oneflow.float32)
    """
    assert input.is_consistent, "input must be a consistent tensor!"
    return ToLocal()(input)
