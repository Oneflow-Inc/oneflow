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
from oneflow.support.blocking import BlockingInfoContext
import oneflow as flow
from oneflow.framework.tensor import register_tensor_op, Tensor
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
        return flow._C.to_consistent(x, placement=placement, sbp=sbp)


@register_tensor_op("to_consistent")
def to_consistent_op(input, placement=None, sbp=None, grad_sbp=None):
    """Cast a local tensor to consistent tensor or cast a
    consistent tensor to another consistent tensor with
    different sbp or placement


    Args:
        input (Tensor): the input tensor.
        placement (flow.placement, optional): the desired placement of returned consistent tensor. Default: if None, the input tensor must be consistent one and use its own placement.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp descriptor of returned consistent tensor. Default: if None, the input tensor must be consistent one and use its own sbp.

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
    assert isinstance(input, Tensor)

    def _check_sbp(sbp):
        if sbp is None:
            pass
        elif isinstance(sbp, (tuple, list)):
            if not all(isinstance(sbp_item, flow.sbp.sbp) for sbp_item in sbp):
                raise TypeError(
                    "sbp parameter must be type of oneflow.sbp.sbp or list/tuple of oneflow.sbp.sbp"
                )
        elif isinstance(sbp, flow.sbp.sbp):
            sbp = (sbp,)
        else:
            raise TypeError(f"Invalid parameter sbp with type {type(sbp)}")

        return sbp

    sbp = _check_sbp(sbp)

    if input.is_consistent:
        # convert consistent tensor to another consistent tensor with different placement or sbp
        if placement is None:
            placement = input.placement

        if sbp is None:
            sbp = input.sbp

        grad_sbp = _check_sbp(grad_sbp)

    else:
        # local tensor to consistent tensor
        if placement is None or sbp is None:
            raise ValueError(
                "Converting a local tensor to consistent tensor must have placement and sbp parameters."
            )

        if not isinstance(placement, flow.placement):
            raise ValueError(f"Invalid parameter placement with type {type(placement)}")

    if grad_sbp is None:
        grad_sbp = tuple()
    with BlockingInfoContext() as ctx:
        return flow._C.to_consistent(input, placement, sbp, grad_sbp)


class ToLocal(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow._C.to_local(x)


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
        >>> input = flow.tensor(np_arr, dtype=flow.float32)
        >>> placement = flow.placement("cpu", {0:range(1)})
        >>> consistent_tensor = input.to_consistent(placement, [flow.sbp.split(0)])
        >>> consistent_tensor.to_local()
        tensor([0.5000, 0.6000, 0.7000], dtype=oneflow.float32)
    """
    assert input.is_consistent, "input must be a consistent tensor!"
    return flow._C.to_local(input)
