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


def print_error_msg():
    msg = ""
    exc_info = sys.exc_info()
    if len(exc_info) > 0:
        msg += str(exc_info[0])
    if len(exc_info) > 1:
        msg += " " + str(exc_info[1])
    print(msg)


def to_torch(flow_tensor):
    r"""
    to_torch(flow_tensor) -> Tensor

    This function is the opposite of :func:`oneflow.utils.from_torch`.

    Args:
        input (oneflow.Tensor): Input Tensor

    Returns:
        torch.Tensor

    For example:

    .. code-block:: python

        import oneflow as flow

        flow_t = flow.tensor([[1, 2, 3], [4, 5, 6]])
        torch_t = flow.utils.to_torch(flow_t)

    This feature ``to_torch`` is at Alpha Stage.
    """
    try:
        import torch
    except:
        print_error_msg()
    assert isinstance(flow_tensor, flow.Tensor)
    assert (
        flow_tensor.is_cuda == False
    ), "Only supports conversion of oneflow tensor whose device is cpu"
    np_data = flow_tensor.cpu().detach().numpy()
    return torch.from_numpy(np_data)
