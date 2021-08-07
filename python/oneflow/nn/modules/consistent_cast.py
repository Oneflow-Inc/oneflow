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
    def __init__(self, placement, sbp):
        super().__init__()
        self.placement = placement
        if isinstance(sbp, flow.sbp.sbp):
            sbp = [sbp]
        for elem in sbp:
            assert isinstance(
                elem, flow.sbp.sbp
            ), "element %s is not an sbp instance" % (sbp)
        self.sbp = sbp

    def forward(self, x, sbp, placement):
        return flow.F.to_consistent(x, placement=placement, sbp=sbp)


@register_tensor_op("to_consistent")
def to_consistent_op(input, placement=None, sbp=None, shape=None):
    """Cast a local tensor to consistent tensor or cast a
    consistent tensor to another consistent tensor with 
    different sbp or placement


    Args:
        input (Tensor): the input tensor.
        placement (flow.placement, optional) – the desired placement of returned consistent tensor. Default: if None, the input tensor must be consistent one and use its own placement.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional) – the desired sbp descriptor of returned consistent tensor. Default: if None, the input tensor must be consistent one and use its own sbp.
        shape (flow.Size, optional) the logical shape of returned consistent tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> placement = flow.placement("cpu", {0:range(1)})
        >>> output_tensor = input.to_consistent(placement, [flow.sbp.split(0)])
        >>> output_tensor.is_consistent
        True
    """
    if isinstance(sbp, flow.sbp.sbp):
        sbp = (sbp,)
    if placement is None or sbp is None:
        assert (
            input.is_consistent
        ), "Converting a local tensor to consistent tensor must have placement and sbp parameters!"
        assert (
            placement is not None or sbp is not None
        ), "Converting a consistent tensor to consistent tensor must have at least one of placement and sbp parameters!"
        placement = input.placement if placement is None else placement
        sbp = input.sbp if sbp is None else sbp
    return flow.F.to_consistent(input, placement, sbp, shape)


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
        >>> consistent_tensor = input.to_consistent(placement, [flow.sbp.split(0)])
        >>> consistent_tensor.to_local()
        tensor([0.5, 0.6, 0.7], dtype=oneflow.float32)
    """
    assert input.is_consistent, "input must be a consistent tensor!"
    return flow.F.to_local(input)
