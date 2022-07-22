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
from typing import List
from oneflow.framework.tensor import Tensor
import oneflow as flow


def pad(
    input: Tensor, pad: List[int], mode: str = "constant", value: float = 0.0
) -> Tensor:
    r"""Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\text{padding\_left}, \text{padding\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom}`
        :math:`\text{padding\_front}, \text{padding\_back})`.

    Padding mode:
        See :class:`torch.nn.ConstantPad2d`, :class:`torch.nn.ReflectionPad2d`, and
        :class:`torch.nn.ReplicationPad2d` for concrete examples on how each of the
        padding modes works. Constant padding is implemented for arbitrary dimensions.
        Replicate and reflection padding is implemented for padding the last 3
        dimensions of 5D input tensor, or the last 2 dimensions of 4D input
        tensor, or the last dimension of 3D input tensor.

    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.

    Args:
        input (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

    Examples::

        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = F.pad(t4d, p2d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 3, 8, 4])
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = F.pad(t4d, p3d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 9, 7, 3])

    """
    assert len(pad) % 2 == 0, "Padding length must be divisible by 2"
    assert len(pad) // 2 <= input.dim(), "Padding length too large"
    if mode == "constant":
        return flow._C.pad(input, pad, mode="constant", value=value)
    else:
        assert (
            value == 0.0
        ), 'Padding mode "{}"" doesn\'t take in value argument'.format(mode)
        if len(pad) == 2 and (input.dim() == 2 or input.dim() == 3):
            if mode == "reflect":
                return flow._C.pad(input, pad, mode="reflect")
            elif mode == "replicate":
                return flow._C.pad(input, pad, mode="reflect")
            elif mode == "circular":
                raise NotImplementedError(
                    "1D circular padding are not supported for now"
                )
            else:
                raise NotImplementedError

        elif len(pad) == 4 and (input.dim() == 3 or input.dim() == 4):
            if mode == "reflect":
                flow._C.pad(input, pad, mode="reflect")
            elif mode == "replicate":
                flow._C.pad(input, pad, mode="replicate")
            elif mode == "circular":
                raise NotImplementedError(
                    "2D circular padding are not supported for now"
                )
            else:
                raise NotImplementedError

        elif len(pad) == 6 and (input.dim() == 4 or input.dim() == 5):
            if mode == "reflect":
                raise NotImplementedError(
                    "3D reflect padding are not supported for now"
                )
            elif mode == "replicate":
                raise NotImplementedError(
                    "3D replicate padding are not supported for now"
                )
            elif mode == "circular":
                raise NotImplementedError(
                    "3D circular padding are not supported for now"
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(
                "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now"
            )
