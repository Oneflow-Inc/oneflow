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
from oneflow.framework.tensor import register_tensor_op
import oneflow._oneflow_internal.lazy_mode as lazy_mode


def _safe_get(list, index, default):
    if index < len(list) and index >= -len(list):
        return list[index]
    else:
        return default


def _parse_args(*args, **kwargs):
    device = None
    dtype = None
    copy = False

    # parse params from args
    if len(args) > 0:
        first_arg = args[0]
        if isinstance(first_arg, flow.Tensor):
            # args format is (another_tensor, copy=False)
            device = first_arg.device
            dtype = first_arg.dtype
            copy = _safe_get(args, 1, False)
        elif isinstance(first_arg, flow.dtype):
            # args format is (dtype, copy=False)
            device = None
            dtype = first_arg
            copy = _safe_get(args, 1, False)
        elif isinstance(first_arg, flow.device):
            # args format is (flow.device, dtype=None, copy=False)
            device = first_arg
            dtype = _safe_get(args, 1, None)
            copy = _safe_get(args, 2, False)
        elif isinstance(first_arg, str):
            # args format is (device_str, dtype=None, copy=False)
            device = first_arg
            dtype = _safe_get(args, 1, None)
            copy = _safe_get(args, 2, False)
        else:
            raise TypeError(f"to() received invalid args {args}")

    # parse params from kwargs
    device = kwargs.get("device", device)
    dtype = kwargs.get("dtype", dtype)
    copy = kwargs.get("copy", copy)

    return device, dtype, copy


def _validate_args(device, dtype, copy, input):
    """
    checks the dtypes of args, and checks if the call to to_op is valid
    """
    if not isinstance(copy, bool):
        raise TypeError("Invalid copy param received: {copy}")

    if not isinstance(dtype, flow.dtype) and dtype is not None:
        raise TypeError("Invalid dtype param received: {dtype}")

    dtype = dtype or input.dtype
    assert isinstance(dtype, flow.dtype), f"Invalid dtype param: {dtype}"

    if input.is_consistent:
        if device is not None and device not in ("cuda", "cpu"):
            raise TypeError(
                "A consistent tensor can only call to() with device_str_without_id, "
                'e.g. to("cuda") or to("cpu"), '
                f"but device param {device} has been received."
            )
    else:
        device = device or input.device
        if isinstance(device, flow.device):
            device = device.type + ":" + str(device.index)

    return device, dtype


@register_tensor_op("to")
def to_op(input, *args, **kwargs):
    """Performs Tensor dtype and/or device conversion.
        A flow.dtype and flow.device are inferred from the arguments of `input.to(*args, **kwargs)`.

    .. note::
        If the ``input`` Tensor already
        has the correct :class:`flow.dtype` and :class:`flow.device`, then ``input`` is returned.
        Otherwise, the returned tensor is a copy of ``input`` with the desired.

    Args:
        input (oneflow.Tensor): An input tensor.
        *args (oneflow.Tensor or oneflow.device or oneflow.dtype): Positional arguments
        **kwargs (oneflow.device or oneflow.dtype) : Key-value arguments

    Returns:
        oneflow.Tensor: A Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> arr = np.random.randint(1, 9, size=(1, 2, 3, 4))
        >>> input = flow.Tensor(arr)
        >>> output = input.to(dtype=flow.float32)
        >>> np.array_equal(arr.astype(np.float32), output.numpy())
        True

    """
    device, dtype, copy = _parse_args(*args, **kwargs)

    device, dtype = _validate_args(device, dtype, copy, input)

    return flow._C.to(input, device, dtype, copy)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
