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
from typing import Union, List

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op


def eye_op(
    n,
    m=None,
    dtype: flow.dtype = flow.float,
    device: Union[str, flow.device] = None,
    placement: flow.placement = None,
    sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    requires_grad: bool = False,
):
    """This operator creates a 2-D Tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n (int): the number of rows.
        m (Optional[int], optional): the number of colums with default being n. Defaults to None.
    
    Keyword args:
        device(flow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: `False`.
    
    Returns:
        oneflow.Tensor: The result Blob with ones on the diagonal and zeros elsewhere.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> out = flow.eye(3, 3)
        >>> out
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]], dtype=oneflow.float32)
    
    """
    if placement is None:
        if isinstance(device, str):
            device = flow.device(device)
        res = flow._C.eye(n, m, dtype=dtype, device=device)
    else:
        assert isinstance(
            placement, flow._oneflow_internal.placement
        ), "placement should be oneflow._oneflow_internal.placement type."
        assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
        if isinstance(sbp, flow.sbp.sbp):
            assert sbp == flow.sbp.broadcast
            sbp = (sbp,)
        else:
            for elem in sbp:
                assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
                assert elem == flow.sbp.broadcast
        assert len(sbp) == len(placement.hierarchy)
        res = flow._C.consistent_eye(n, m, dtype=dtype, placement=placement, sbp=sbp)

    res.requires_grad = requires_grad
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
