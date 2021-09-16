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
from oneflow.framework.tensor import Tensor
from oneflow.framework.tensor import register_tensor_op
from typing import Any, TypeVar
from oneflow.nn.module import Module


def norm_except_dim(v: Tensor, pow: int, dim: int):
    if dim == -1:
        return flow.linalg.norm(v, "fro")
    elif dim == 0:
        output_size = [1] * v.dim()
        output_size[0] = v.size(0)
        return flow.linalg.norm(v.view(v.size(0), -1), ord=2, dim=1).view(*output_size)
    elif dim == v.dim() - 1:
        output_size = [1] * v.dim()
        output_size[v.dim() - 1] = v.size(v.dim() - 1)
        return flow.linalg.norm(v.view(-1, v.size(v.dim() - 1)), ord=2, dim=0).view(
            *output_size
        )
    else:
        return norm_except_dim(
            flow.transpose(flow.transpose(v, 0, dim), pow, 0), 0, dim
        )


class WeightNorm(object):
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module: Module) -> Any:
        g = getattr(module, self.name + "_g")
        v = getattr(module, self.name + "_v")
        return v * (g / norm_except_dim(v, 2, self.dim))

    @staticmethod
    def apply(module, name: str, dim: int) -> "WeightNorm":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two weight_norm hooks on "
                    "the same parameter {}".format(name)
                )

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(
            name + "_g", flow.nn.Parameter(norm_except_dim(weight, 2, dim).data)
        )
        module.register_parameter(name + "_v", flow.nn.Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar("T_module", bound=Module)


def weight_norm(module: T_module, name: str = "weight", dim: int = 0) -> T_module:
    r"""Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(flow.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim)
    return module


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
