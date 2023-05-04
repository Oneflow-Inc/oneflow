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

# The file mostly copyed from https://github.com/pytorch/pytorch/blob/master/torch/cuda/amp/autocast_mode.py
import oneflow as flow
import functools
import collections

try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
from typing import Any

string_classes = (str, bytes)

__all__ = ["autocast", "custom_fwd", "custom_bwd"]
from typing import Any, Optional


class autocast(flow.amp.autocast_mode.autocast):
    r"""
    See :class:`oneflow.autocast`.
    ``oneflow.cuda.amp.autocast(args...)`` is equivalent to ``oneflow.autocast("cuda", args...)``
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: Optional[flow.dtype] = None,
        cache_enabled: Optional[bool] = None,
    ):
        super().__init__(
            "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        return super().__call__(func)


# Casts Tensors and containers of Tensors.  Special-cases passthroughs for strings and np.ndarrays, which
# may be falsely detected as "Iterables."
def _cast(value, dtype):
    if isinstance(value, flow.Tensor):
        is_eligible = (
            value.is_floating_point()
            and value.is_cuda
            and (value.dtype is not flow.float64)
        )
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, string_classes):
        return value
    elif HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, collections.abc.Iterable):
        iterable = map(lambda v: _cast(v, dtype), value)
        if isinstance(value, list) or isinstance(value, tuple):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value


# custom_fwd is a decorator that may or may not be used with arguments, following
# https://github.com/dabeaz/python-cookbook/tree/master/src/9/defining_a_decorator_that_takes_an_optional_argument.
# this works:
#     @custom_fwd
#     def forward(...):
# this also works:
#     @custom_fwd(cast_inputs=flow.float)
#     def forward(...):
def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    Helper decorator for ``forward`` methods of custom autograd functions (subclasses of
    :class:`flow.autograd.Function`).  See the :ref:`example page<amp-custom-examples>` for more detail.
    Args:
        cast_inputs (:class:`flow.dtype` or None, optional, default=None):  If not ``None``,
            when ``forward`` runs in an autocast-enabled region, casts incoming
            floating-point CUDA Tensors to the target dtype (non-floating-point Tensors are not affected),
            then executes ``forward`` with autocast disabled.
            If ``None``, ``forward``'s internal ops execute with the current autocast state.
    .. note::
        If the decorated ``forward`` is called outside an autocast-enabled region,
        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
    """
    if fwd is None:
        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        args[0]._dtype = flow.get_autocast_gpu_dtype()
        if cast_inputs is None:
            args[0]._fwd_used_autocast = flow.is_autocast_enabled()
            return fwd(*args, **kwargs)
        else:
            autocast_context = flow.is_autocast_enabled()
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with autocast(enabled=False):
                    return fwd(*_cast(args, cast_inputs), **_cast(kwargs, cast_inputs))
            else:
                return fwd(*args, **kwargs)

    return decorate_fwd


# Autograd ensures incoming gradients are the same type as forward outputs.  Allowing a separate
# cast_inputs argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs supplied to custom_fwd.
def custom_bwd(bwd):
    """
    Helper decorator for backward methods of custom autograd functions (subclasses of
    :class:`flow.autograd.Function`).
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.
    """

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(enabled=args[0]._fwd_used_autocast, dtype=args[0]._dtype):
            return bwd(*args, **kwargs)

    return decorate_bwd
