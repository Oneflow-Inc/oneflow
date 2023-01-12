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

import warnings
from typing import Union, Iterable

import numpy as np
import oneflow as flow

from oneflow.framework.tensor import Tensor
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module


_tensor_or_tensors = Union[Tensor, Iterable[Tensor]]


def clip_grad_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Tensor:
    r"""Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Parameters after cliping gradient norm
        Total norm of the parameters (viewed as a single vector).
    

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.tensor(np.array([[2, 3, 4], [1.5, 2.6, 3.7]]).astype(np.float32), requires_grad=True)
        >>> m1 = flow.nn.ReLU()
        >>> out1 = m1(x1)
        >>> out1 = out1.sum()
        >>> out1.backward()
        >>> norm1 = flow.nn.utils.clip_grad_norm_(x1, 0.6, 1.0)
        >>> norm1
        tensor(6., dtype=oneflow.float32)
        >>> x1.grad
        tensor([[0.1000, 0.1000, 0.1000],
                [0.1000, 0.1000, 0.1000]], dtype=oneflow.float32)
        >>> x2 = flow.tensor(np.array([[-2, -3, -4], [2.5, 0, 3.2]]).astype(np.float32), requires_grad=True)
        >>> out2 = flow.atan(x2)
        >>> out2 = out2.sum()
        >>> out2.backward()
        >>> norm2 = flow.nn.utils.clip_grad_norm_(x2, 0.5)
        >>> norm2
        tensor(1.0394, dtype=oneflow.float32)
        >>> x2.grad
        tensor([[0.0962, 0.0481, 0.0283],
                [0.0663, 0.4810, 0.0428]], dtype=oneflow.float32)

    """

    if isinstance(parameters, (Tensor, flow._oneflow_internal.Tensor)):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return flow.tensor(0.0)

    if parameters[0].is_global:
        assert all(
            [p.is_global for p in parameters]
        ), "All parameters must be global tensor."
        sbp_broadcast = [flow.sbp.broadcast for _ in parameters[0].sbp]
        param0_placement = parameters[0].placement
        if norm_type == float("inf"):
            norms = [
                p.grad.detach()
                .to_global(sbp=sbp_broadcast)
                .abs()
                .max()
                .to_global(placement=param0_placement)
                for p in parameters
            ]
            total_norm = norms[0] if len(norms) == 1 else flow.max(flow.stack(norms))
        elif norm_type == float("-inf"):
            norms = [
                p.grad.detach()
                .to_global(sbp=sbp_broadcast)
                .abs()
                .min()
                .to_global(placement=param0_placement)
                for p in parameters
            ]
            total_norm = norms[0] if len(norms) == 1 else flow.min(flow.stack(norms))
        else:
            total_norm = flow.linalg.vector_norm(
                flow.stack(
                    [
                        flow.linalg.vector_norm(
                            p.grad.detach().to_global(sbp=sbp_broadcast), norm_type
                        ).to_global(placement=param0_placement)
                        for p in parameters
                    ]
                ),
                norm_type,
            )
        if error_if_nonfinite and flow.logical_or(
            total_norm.isnan(), total_norm.isinf()
        ):
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, "
                "set `error_if_nonfinite=False`"
            )
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = clip_coef.clamp(max=1.0)
        for p in parameters:
            p.grad.detach().mul_(clip_coef_clamped.to_global(placement=p.placement))
    else:
        device = parameters[0].grad.device
        if norm_type == float("inf"):
            norms = [p.grad.detach().abs().max().to(device) for p in parameters]
            total_norm = norms[0] if len(norms) == 1 else flow.max(flow.stack(norms))
        elif norm_type == float("-inf"):
            norms = [p.grad.detach().abs().min().to(device) for p in parameters]
            total_norm = norms[0] if len(norms) == 1 else flow.min(flow.stack(norms))
        else:
            total_norm = flow.linalg.vector_norm(
                flow.stack(
                    [
                        flow.linalg.vector_norm(p.grad.detach(), norm_type).to(device)
                        for p in parameters
                    ]
                ),
                norm_type,
            )
        if error_if_nonfinite and flow.logical_or(
            total_norm.isnan(), total_norm.isinf()
        ):
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, "
                "set `error_if_nonfinite=False`"
            )
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = clip_coef.clamp(max=1.0)
        for p in parameters:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return total_norm


def clip_grad_value_(parameters: _tensor_or_tensors, clip_value: float) -> None:
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
    """
    if isinstance(parameters, flow.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        # TODO: Switch to inplace clamp function
        p.grad[:] = p.grad.clamp(min=-clip_value, max=clip_value)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
