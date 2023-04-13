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

from oneflow.cuda.type_tensor import *
from oneflow.cuda._utils import _get_device_index

from typing import Optional, Tuple, Union, Any


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    # This function never throws and returns 0 if driver is missing or can't
    # be initialized
    return device_count() > 0


def device_count() -> int:
    r"""Returns the number of GPUs available."""
    return flow._oneflow_internal.CudaGetDeviceCount()


def current_device() -> int:
    r"""Returns local rank as device index."""
    return flow._oneflow_internal.GetCudaDeviceIndex()


def get_device_properties(device: Union[flow.device, str, int] = None):
    r"""Gets the properties of a device.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.cuda.get_device_properties.html.

    Args:
        device(oneflow.device or str or int): device for which to return the properties of the device.

    Returns:
        the properties of the device.
    """
    device = _get_device_index(device, optional=True)
    return flow._oneflow_internal._get_device_properties(device)


def get_device_capability(
    device: Optional[Union[flow.device, str, int]] = None
) -> Tuple[int, int]:
    r"""Gets the cuda capability of a device.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.cuda.get_device_capability.html.

    Args:
        device (oneflow.device or int or str, optional): device for which to return the
            device capability. It uses the current device, given by
            :func:`~oneflow.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    device_prop = get_device_properties(device)
    return device_prop.major, device_prop.minor


def get_device_name(device: Optional[Union[flow.device, str, int]] = None) -> str:
    r"""Gets the name of a device.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.cuda.get_device_name.html.

    Args:
        device (oneflow.device or int or str, optional): device for which to return the
            name. It uses the current device, given by :func:`~oneflow.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


def manual_seed_all(seed) -> None:
    r"""Sets the seed for generating random numbers on all GPUs.
    
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.cuda.manual_seed_all.html.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)
    flow._oneflow_internal.ManualSeedAllCudaGenerator(seed)


def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers for the current GPU.
    
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.cuda.manual_seed.html.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)
    idx = current_device()
    flow._oneflow_internal.manual_seed(seed, "cuda", idx)


def set_device(device: Union[flow.device, str, int]) -> None:
    r"""Sets the current device.
    
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.cuda.set_device.html.

    Usage of this function is discouraged in favor of :attr:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (flow.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device_idx = _get_device_index(device)
    if device_idx < 0:
        return
    if flow.env.get_world_size() > 0:
        if device_idx == flow.env.get_local_rank():
            return
        raise ValueError(
            "Setting cuda device to a device whose index does not equal to the local rank is not supported."
        )
    flow._oneflow_internal.SetCudaDeviceIndex(device_idx)


def synchronize(device: Union[flow.device, str, int, None] = None) -> None:
    r"""
    
    Waits for all kernels in all streams on a CUDA device to complete.
    
    Note:
        In the eager mode of oneflow, all operations will be converted
        into instructions executed in the virtual machine, 
        so in order to comply with the semantics of synchronization,
        this function will call the `eager.Sync()` function before the device is synchronized,
        which may affect the operations executed in other devices.

    Args:
        device (flow.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~oneflow.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device_idx = _get_device_index(device, optional=True)
    if device_idx >= 0:
        flow._oneflow_internal.eager.Sync()
        flow._oneflow_internal.CudaSynchronize(device_idx)


def empty_cache() -> None:
    r"""
    
    Releases all unoccupied cached memory currently held by the caching 
    allocators of all OneFlow streams so those can be re-allocated in OneFlow streams 
    or other GPU application and visible in `nvidia-smi`.
    
    Note:
            :func:`~flow.cuda.empty_cache` may enable one stream to release memory 
            and then freed memory can be used by another stream. It may also help reduce 
            fragmentation of GPU memory in certain cases.

    """
    return flow._oneflow_internal.EmptyCache()


from .random import *  # noqa: F403


class Event:
    def __init__(self):
        raise NotImplementedError()
