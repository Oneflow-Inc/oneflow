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
# This file is mostly copied from PyTorch

import oneflow as flow
from typing import Callable, Any


class saved_tensors_hooks:
    """Context-manager that sets a pair of pack / unpack hooks for saved tensors.

    Use this context-manager to define how intermediary results of an operation
    should be packed before saving, and unpacked on retrieval.

    In that context, the ``pack_hook`` function will be called everytime an
    operation saves a tensor for backward (this includes intermediary results
    saved using
    :func:`~oneflow.autograd.function.save_for_backward` but
    also those recorded by a OneFlow-defined operation). The output of
    ``pack_hook`` is then stored in the computation graph instead of the
    original tensor.

    The ``unpack_hook`` is called when the saved tensor needs to be accessed,
    namely when executing :func:`oneflow.Tensor.backward()` or
    :func:`oneflow.autograd.grad()`. It takes as argument the *packed* object
    returned by ``pack_hook`` and should return a tensor which has the same
    content as the original tensor (passed as input to the corresponding
    ``pack_hook``).

    The hooks should have the following signatures:

        pack_hook(tensor: Tensor) -> Any

        unpack_hook(Any) -> Tensor

    where the return value of ``pack_hook`` is a valid input to ``unpack_hook``.

    In general, you want ``unpack_hook(pack_hook(t))`` to be equal to ``t`` in terms
    of value, size, dtype and device.

    Example::

        >>> def pack_hook(x):
        ...     print("Packing", x)
        ...     return x
        >>>
        >>> def unpack_hook(x):
        ...     print("Unpacking", x)
        ...     return x
        >>>
        >>> a = flow.ones(5, requires_grad=True)
        >>> b = flow.ones(5, requires_grad=True) * 2
        >>> with flow.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        ...     y = a * b
        Packing tensor([1., 1., 1., 1., 1.])
        Packing tensor([2., 2., 2., 2., 2.])
        >>> y.sum().backward()
        Unpacking tensor([1., 1., 1., 1., 1.])
        Unpacking tensor([2., 2., 2., 2., 2.])

    .. warning ::
        Performing an inplace operation on the input to either hooks may lead
        to undefined behavior.

    .. warning ::
        Only one pair of hooks is allowed at a time. When recursively nesting this
        context-manager, only the inner-most pair of hooks will be applied.
    """

    def __init__(
        self,
        pack_hook: Callable[["flow.Tensor"], Any],
        unpack_hook: Callable[[Any], "flow.Tensor"],
    ):
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self):
        flow._oneflow_internal.autograd.graph.append_new_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args: Any):
        flow._oneflow_internal.autograd.graph.pop_hooks()
