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

from oneflow._oneflow_internal import TensorTuple
from oneflow._oneflow_internal.autograd import AutogradFunctionBase


class Function(AutogradFunctionBase):
    r"""
    Function(self)

    Base class to create custom autograd.Function.

    To create a custom autograd.Function, subclass this class and implement the ``forward()``
    and ``backward()`` static methods. Then, to use your custom op in the forward pass, call the
    class method ``apply()`` or ``__call__()``. Do not call ``forward()`` directly.

    For example:

    .. code-block:: python

        class Exp(Function):
            @staticmethod
            def forward(ctx, i):
                result = i.exp()
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                result, = ctx.saved_tensors
                return grad_output * result

        # Use it by calling the apply method or __call__ method
        output = Exp.apply(input)  # output = Exp()(input)
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *inputs):
        r"""
        See :meth:`self.apply`.
        """
        return self.apply(*inputs)

    @classmethod
    def apply(cls, *inputs):
        r"""
        Calculate output tensors and build backward graph.
        """
        return AutogradFunctionBase.apply(
            cls.__name__, cls.forward, cls.backward, *inputs
        )

    @staticmethod
    def forward(ctx, *inputs):
        r"""
        Override this function for custom forward calculation.
        """
        raise NotImplementedError(
            "You must implement the forward function for custom autograd.Function."
        )

    @staticmethod
    def backward(ctx, *out_grads):
        r"""
        Override this function for custom backward calculation.
        """
        raise NotImplementedError(
            "You must implement the backward function for custom autograd.Function."
        )
