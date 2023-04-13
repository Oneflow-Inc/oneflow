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
import functools
import warnings
from typing import Any, Optional

import oneflow as flow
import oneflow._oneflow_internal.lazy_mode as lazy_mode


__all__ = ["autocast_decorator", "autocast"]


def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    return decorate_autocast


class autocast(object):
    r"""
    Note:
      The following doc was origined by pytorch, see
      https://github.com/pytorch/pytorch/blob/master/torch/amp/autocast_mode.py#L19-L179

    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

    Example for CUDA Devices::

        # Creates model and optimizer in default precision
        model = Net().cuda()
        optimizer = optim.SGD(model.parameters(), ...)

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with oneflow.autocast(device_type="cuda"):
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            loss.backward()
            optimizer.step()


    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...
            @oneflow.autocast(device_type="cuda")
            def forward(self, input):
                ...

    Floating-point Tensors produced in an autocast-enabled region may be ``float16``.
    After returning to an autocast-disabled region, using them with floating-point
    Tensors of different dtypes may cause type mismatch errors.  If so, cast the Tensor(s)
    produced in the autocast region back to ``float32`` (or other dtype if desired).
    If a Tensor from the autocast region is already ``float32``, the cast is a no-op,
    and incurs no additional overhead.
    CUDA Example::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = oneflow.rand((8, 8), device="cuda")
        b_float32 = oneflow.rand((8, 8), device="cuda")
        c_float32 = oneflow.rand((8, 8), device="cuda")
        d_float32 = oneflow.rand((8, 8), device="cuda")

        with oneflow.autocast(device_type="cuda"):
            # oneflow.mm is on autocast's list of ops that should run in float16.
            # Inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            e_float16 = oneflow.mm(a_float32, b_float32)
            # Also handles mixed input types
            f_float16 = oneflow.mm(d_float32, e_float16)

        # After exiting autocast, calls f_float16.float() to use with d_float32
        g_float32 = oneflow.mm(d_float32, f_float16.float())

    CPU Training Example::

        # Creates model and optimizer in default precision
        model = Net()
        optimizer = optim.SGD(model.parameters(), ...)

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()

                # Runs the forward pass with autocasting.
                with oneflow.autocast(device_type="cpu", dtype=oneflow.bfloat16):
                    output = model(input)
                    loss = loss_fn(output, target)

                loss.backward()
                optimizer.step()


    CPU Inference Example::

        # Creates model in default precision
        model = Net().eval()

        with oneflow.autocast(device_type="cpu", dtype=oneflow.bfloat16):
            for input in data:
                # Runs the forward pass with autocasting.
                output = model(input)

    The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.

    Args:
        device_type(str, required):  Whether to use 'cuda' or 'cpu' device
        enabled(bool, optional):  Whether autocasting should be enabled in the region.
            Default: ``True``
        dtype(oneflow_dtype, optional):  Whether to use oneflow.float16 or oneflow.bfloat16.
        cache_enabled(bool, optional):  Whether the weight cache inside autocast should be enabled.
            Default: ``True``
    """

    def __init__(
        self,
        device_type: str,
        dtype: Optional[flow.dtype] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        self.device = device_type
        if self.device == "cuda":
            self.fast_dtype = flow.get_autocast_gpu_dtype()
        elif self.device == "cpu":
            self.fast_dtype = flow.get_autocast_cpu_dtype()
        else:
            raise RuntimeError(
                "User specified autocast device_type must be 'cuda' or 'cpu'"
            )
        self.cache_enabled = flow.is_autocast_cache_enabled()

        if dtype is not None:
            self.fast_dtype = dtype
        if cache_enabled is not None:
            self.cache_enabled = cache_enabled

        if self.device == "cpu":
            warnings.warn(
                "CPU autocast is not supported currently. Disabling autocast."
            )
            enabled = False
        if lazy_mode.is_enabled():
            warnings.warn(
                "Autocast is not supported for lazy mode. Disabling autocast."
            )
            enabled = False
        self.enabled = enabled

    def __enter__(self):
        self.autocast_mode = flow._oneflow_internal.AutoCastMode(
            self.device, self.fast_dtype, self.enabled, self.cache_enabled
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        del self.autocast_mode

    def __call__(self, func):
        return autocast_decorator(self, func)
