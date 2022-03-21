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


def to_torch(flow_tensor):
    r"""
    This function is the opposite of :func:`oneflow.utils.from_torch`.

    .. code-block:: python

        import oneflow as flow

        flow_t = flow.tensor([[1, 2, 3], [4, 5, 6]])
        torch_t = flow.utils.to_torch(flow_t)

    """
    try:
        import torch
    except:
        print("No module named torch")
    assert isinstance(flow_tensor, flow.Tensor)
    assert (
        flow_tensor.is_cuda == False
    ), "Only supports conversion of tensor whose device is cpu"
    np_data = flow_tensor.cpu().detach().numpy()
    # assert (
    #     torch.from_numpy(np_data).data_ptr() == np_data.__array_interface__["data"][0]
    # )
    return torch.from_numpy(np_data)
