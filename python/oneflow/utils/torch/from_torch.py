import torch
import oneflow as flow
from oneflow._C import from_numpy as flow_from_numpy

def from_torch(torch_tensor):
    assert isinstance(torch_tensor, torch.Tensor)
    np_data = torch_tensor.cpu().detach().numpy()
    device = torch_tensor.device.__str__()
    return flow_from_numpy(np_data).to(device=device)