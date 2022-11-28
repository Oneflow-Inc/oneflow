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
from typing import Any, Optional


def _get_current_device_index() -> int:
    r"""Checks if there are CUDA devices available and
    returns the device index of the current default CUDA device.
    Returns -1 in case there are no CUDA devices available.
    
    Arguments: ``None``
    """
    if flow.cuda.is_available():
        return flow.cuda.current_device()
    return -1


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a flow.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``flow.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    device_idx: Optional[int] = None
    if isinstance(device, str):
        device = flow.device(device)
    if isinstance(device, flow.device):
        if allow_cpu:
            if device.type not in ["cuda", "cpu"]:
                raise ValueError(
                    "Expected a cuda or cpu device, but got: {}".format(device)
                )
        elif device.type != "cuda":
            raise ValueError("Expected a cuda device, but got: {}".format(device))
        device_idx = -1 if device.type == "cpu" else device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        if optional:
            device_idx = _get_current_device_index()
        else:
            raise ValueError(
                "Expected a flow.device with a specified index "
                "or an integer, but got:{}".format(device)
            )
    return device_idx
