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
from typing import List, Optional, Union
import math
import oneflow as flow


def linspace_op(
    start: float,
    end: float,
    steps: int,
    dtype: flow.dtype = flow.float32,
    device: Union[str, flow.device] = None,
    placement: flow.placement = None,
    sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    requires_grad: bool = False,
):
    r"""
    Creates a one-dimensional tensor of size :attr:`steps` whose values are evenly
    spaced from :attr:`start` to :attr:`end`, inclusive. That is, the value are:

    .. math::
        (\text{start},
        \text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1},
        \ldots,
        \text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1},
        \text{end})

    Args:
        start (float): the starting value for the set of points
        end (float): the ending value for the set of points
        steps (int): size of the constructed tensor

    Keyword arguments:
        dtype(flow.dtype, optional): If `dtype` is not given, the `dtype` is inferred to be `flow.float32`.
        device(flow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: `False`.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> y = flow.linspace(3, 10, steps=5)
        >>> y
        tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000], dtype=oneflow.float32)

    """
    step = 1.0
    if steps == 0:
        end = start
    elif steps == 1:
        end = start + 1.0
    else:
        step = (end - start) * 1.0 / (steps - 1)
        if math.isclose(((end - start) / (steps - 1)) * (steps - 1), (end - start)):
            end = end + step / 2.0
    if placement is None:
        if isinstance(device, str):
            device = flow.device(device)
        res = flow._C.arange(start, end, step, dtype=dtype, device=device)
    else:
        assert isinstance(
            placement, flow._oneflow_internal.placement
        ), "placement should be oneflow._oneflow_internal.placement type."
        assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
        if isinstance(sbp, flow.sbp.sbp):
            sbp = (sbp,)
        else:
            for elem in sbp:
                assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
        assert len(sbp) == len(placement.hierarchy)
        res = flow._C.consistent_arange(
            start, end, step, dtype=dtype, placement=placement, sbp=sbp
        )

    res.requires_grad = requires_grad
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
