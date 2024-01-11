import torch
import oneflow as flow
from typing import List, Dict, Any

def parse_device(args: List[Any], kwargs: Dict[str, Any]):
    if "device" in kwargs:
        return kwargs['device']
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
