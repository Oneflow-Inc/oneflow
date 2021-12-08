import oneflow as flow 
import torch 
import numpy as np

#arr = np.arange(24).reshape(2,2,2,3)
arr = np.arange(27).reshape(3,3,3)
print("input  >>>>>>>>>>> \n", arr)

x_flow = flow.tensor(arr)
x_torch = torch.tensor(arr)

# y_flow = x_flow[0:1, 1:2, 0:2, 1:2]
# y_torch = x_torch[0:1, 1:2, 0:2, 1:2]
# y_numpy = arr[0:1, 1:2, 0:2, 1:2]
y_flow = x_flow[1:2, 1:3, 1:2]
y_torch = x_torch[1:2, 1:3, 1:2]
y_numpy = arr[1:2, 1:3, 1:2]

print("flow output >>>>>>>>>>> \n", y_flow.numpy())
print("torch output >>>>>>>>>>> \n", y_torch.numpy())
print("numpy output >>>>>>>>>>> \n", y_numpy)

print("\nflow input stride >>>>> ", x_flow.stride())
print("flow output stride >>>>> ", y_flow.stride())
print("flow output offset >>>>> ", y_flow.storage_offset())

print("\ntorch input stride >>>>> ", x_torch.stride())
print("torch output stride >>>>> ", y_torch.stride())
print("torch output offset >>>>> ", y_torch.storage_offset())