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
from typing import List, Union


def _checkpoint_without_reentrant(function, *args):
    """Checkpointining without re-entrant autograd
    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        *args: Arguments to pass in to the given ``function``.
    """

    storage: List[Union[flow.Tensor, None]] = []
    counter = 0

    def pack(x):
        nonlocal counter
        counter += 1
        return counter - 1

    # TODO(jianhao): support restoring rng state once we have flow.random.fork_rng
    def unpack(x):
        if len(storage) == 0:

            def inner_pack(inner):
                storage.append(inner)
                return None

            def inner_unpack(packed):
                raise RuntimeError(
                    "You are calling backwards on a tensor that is never exposed. Please open an issue."
                )

            with flow.enable_grad():
                with flow.autograd.graph.saved_tensors_hooks(inner_pack, inner_unpack):
                    _unused = function(*args)

        return storage[x]

    with flow.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = function(*args)

    return output


def checkpoint(function, *args):
    r"""Checkpoint a model or part of the model

    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.

    Specifically, in the forward pass, :attr:`function` will run in
    :func:`flow.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retrieved, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.

    The output of :attr:`function` can contain non-Tensor values and gradient
    recording is only performed for the Tensor values. Note that if the output
    consists of nested structures (ex: custom objects, lists, dicts etc.)
    consisting of Tensors, these Tensors nested in custom structures will not
    be considered as part of autograd.


    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.

    .. warning::
        Preserving rng states is not supported now, so that the behavior of
        checkpointing does not fully align with PyTorch.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    return _checkpoint_without_reentrant(function, *args)
