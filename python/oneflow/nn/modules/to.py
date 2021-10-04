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


def _tensor_to(input, device, dtype, copy=False):
    assert input.is_local

    device = device or input.device
    assert isinstance(device, flow.device), f"Invalid device param: {device}"
    dtype = dtype or input.dtype
    assert isinstance(dtype, flow.dtype), f"Invalid dtype param: {dtype}"

    ret = input
    copy_happened = False
    if device != ret.device:
        ret = flow._C.copy(ret, device_type=device.type, device_id=device.index)
        copy_happened = True

    if dtype != ret.dtype:
        ret = flow._C.cast(ret, dtype=dtype)
        copy_happened = True

    if copy and not copy_happened:
        ret = flow._C.copy(ret, device_type=ret.device.type, device_id=ret.device.index)

    return ret


def _consistent_tensor_to(input, device_type, dtype, copy=False):
    assert input.is_consistent
    # TODO(zwx): support lazy check_meta_consistency
    # input.check_meta_consistency()

    device_type = device_type or input.placement.device_type
    assert isinstance(device_type, str)

    dtype = dtype or input.dtype
    assert isinstance(dtype, flow.dtype)

    if device_type == input.placement.device_type and dtype == input.dtype:
        return input if not copy else input.clone()

    if lazy_mode.is_enabled():
        return _lazy_consistent_tensor_to(input, device_type, dtype)
    else:
        return _eager_consistent_tensor_to(input, device_type, dtype)


def _lazy_consistent_tensor_to(input, device_type, dtype):
    ret = input

    if dtype != ret.dtype:
        ret = flow._C.cast(ret, dtype=dtype)

    if device_type != ret.placement.device_type:
        ret = flow._C.copy(ret, device_type=device_type, device_id=0)

    return ret


def _eager_consistent_tensor_to(input, device_type, dtype):
    input.check_meta_consistency()

    if device_type == input.placement.device_type and dtype != input.dtype:
        return flow._C.cast(input, dtype=dtype)
    device = flow.device(device_type)
    placement = flow._oneflow_internal._ReplacePlacementDeviceTag(
        input.placement, device_type
    )
    sbp = input.sbp

    local_input = input.to_local()
    local_output = _tensor_to(local_input, device, dtype, False)
    return local_output.to_consistent(placement=placement, sbp=sbp)


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
        if device is not None and device not in ("cuda", "cpu"):
            raise TypeError(
                "A consistent tensor can only call to() with device_str_without_id, "
                'e.g. to("cuda") or to("cpu"), '
                f"but device param {device} has been received."
            )

        return _consistent_tensor_to(input, device, dtype, copy=copy)
    else:
        if isinstance(device, str):
            device = flow.device(device)

        return _tensor_to(input, device, dtype, copy)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
