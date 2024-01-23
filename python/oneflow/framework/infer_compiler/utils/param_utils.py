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
from typing import Any, Dict, List

import oneflow as flow
import torch


def parse_device(args: List[Any], kwargs: Dict[str, Any]):
    if "device" in kwargs:
        return kwargs["device"]
    for x in args:
        if isinstance(x, (flow.device, torch.device)):
            return x
        if x in ["cpu", "cuda"]:
            return x
    return None


def check_device(current_device, target_device) -> bool:
    def _convert(device):
        assert isinstance(device, (str, torch.device, flow.device))
        if isinstance(device, torch.device):
            index = device.index if device.index is not None else 0
            return flow.device(device.type, index)
        if isinstance(device, str):
            return flow.device(device)
        return device

    return _convert(current_device) == _convert(target_device)
