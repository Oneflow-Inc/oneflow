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


def _tensor_to(input, device=None, dtype=None, copy=False):
    ret = input

    if device is None:
        assert input.is_local, "input tensor must be local when param device is None"
        device = input.device

    if dtype is None:
        dtype = input.dtype

    assert isinstance(device, flow.device), f"Invalid device param: {device}"

    if (device != input.device) or copy:
        ret = flow.F.copy(ret, device_type=device.type, device_id=device.index)

    if (dtype != input.dtype) or copy:
        ret = flow.F.cast(ret, dtype=dtype)

    return ret


def _consistent_tensor_to(input, device):
    assert input.is_consistent
    assert isinstance(device, str)

    # the same device as input
    if device == input.placement.device_type:
        return input

    # NOTE(zwx): eager consistent interpreter for copy is not ready
    # return flow.F.copy(input, device_type=device, device_id=0)

    # NOTE(zwx): eager consistent interpreter for to_consistent is not ready too
    out_placement = flow._oneflow_internal._ReplacePlacementDeviceTag(
        input.placement, device
    )
    sbp = input.sbp
    return input.to_consistent(out_placement, sbp)


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

    if not isinstance(dtype, flow.dtype) and dtype is not None:
        raise TypeError("Invalid dtype param received: {dtype}")

    if not isinstance(copy, bool):
        raise TypeError("Invalid copy param received: {copy}")

    if input.is_consistent:
        if device not in ("cuda", "cpu"):
            raise TypeError(
                "A consistent tensor can only call to() with device_str_without_id, "
                'e.g. to("cuda") or to("cpu"), '
                f"but device param {device} has been received."
            )

        if dtype is not None:
            raise TypeError(
                "Can not call to() for a consistent tensor with dtype param"
            )

        if not input.is_lazy:
            input.check_meta_consistency()

        return _consistent_tensor_to(input, device)
    else:
        if isinstance(device, str):
            device = flow.device(device)

        return _tensor_to(input, device, dtype, copy)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
