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
import sys
import oneflow as flow
from oneflow._C import from_numpy as flow_from_numpy


def print_error_msg():
    msg = ""
    exc_info = sys.exc_info()
    if len(exc_info) > 0:
        msg += str(exc_info[0])
    if len(exc_info) > 1:
        msg += " " + str(exc_info[1])
    print(msg)


def from_torch(torch_tensor):
    r"""
    from_torch(torch_tensor) -> Tensor

    Create a oneflow tensor from torch tensor.

    The returned tensor and torch tensor share the same memory. 
    
    .. note::
        This function can be used in special data processing stages, torch's some cpu ops can be used. 

    Args:
        input (torch.Tensor): Input Tensor

    Returns:
        oneflow.Tensor

    For example:

    .. code-block:: python

        import oneflow as flow
        import torch

        torch_t = torch.tensor([[1, 2, 3], [4, 5, 6]])
        flow_t = flow.utils.tensor.from_torch(torch_t)
    
    This feature ``from_torch`` is at Alpha Stage.
    """
    try:
        import torch
    except:
        print_error_msg()
    assert isinstance(torch_tensor, torch.Tensor)
    return flow.from_dlpack(torch.to_dlpack(torch_tensor))


def to_torch(flow_tensor):
    r"""
    to_torch(flow_tensor) -> Tensor

    Create a torch tensor from oneflow tensor.

    The returned tensor and oneflow tensor share the same memory. 
    
    .. note::
        Currently only local tensor is supported.

    Args:
        input (oneflow.Tensor): Input Tensor

    Returns:
        torch.Tensor

    For example:

    .. code-block:: python

        import oneflow as flow
        import torch

        flow_t = flow.tensor([[1, 2, 3], [4, 5, 6]])
        torch_t = flow.utils.tensor.to_torch(flow_t)

    This feature ``to_torch`` is at Alpha Stage.
    """
    try:
        import torch
    except:
        print_error_msg()
    assert isinstance(flow_tensor, flow.Tensor)
    if flow_tensor.is_global:
        print(
            "WARNING: `to_torch` received a global tensor. A PyTorch CPU tensor which is a copy of its data will be returned."
        )
        return torch.from_numpy(flow_tensor.numpy())
    return torch.from_dlpack(flow.to_dlpack(flow_tensor))
