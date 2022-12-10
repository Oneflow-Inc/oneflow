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
from oneflow.npu._utils import _get_device_index
from typing import Union, Any


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    # This function never throws and returns 0 if driver is missing or can't
    # be initialized
    return device_count() > 0


def device_count() -> int:
    r"""Returns the number of GPUs available."""
    return flow._oneflow_internal.NpuGetDeviceCount()

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
        flow._oneflow_internal.NpuSynchronize(device_idx)

def current_device() -> int:
    r"""Returns local rank as device index."""
    return flow._oneflow_internal.GetNpuDeviceIndex()
