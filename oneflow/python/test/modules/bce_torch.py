import torch
import torch.nn as nn
import numpy as np

x = np.array([[1.2, 0.2, -0.3], [0.7, 0.6, -2]]).astype(np.float32)
y = np.array([[0, 1, 0], [1, 0, 1]]).astype(np.float32)
w = np.array([[2, 2, 2], [2, 2, 2]]).astype(np.float32)

input_torch = torch.from_numpy(x)
input_torch.requires_grad_()
target_torch = torch.from_numpy(y)
weight_torch = torch.from_numpy(w)

m = nn.Sigmoid()
loss = nn.BCELoss(weight_torch, reduction="none")
output = loss(m(input_torch), target_torch)

output = output.sum()
output.backward()

print("input_torch.grad.numpy():")
print(input_torch.grad.numpy())

# [[ 1.5370497  -0.90033215  0.851115  ]
#  [-0.6636245   1.2913125  -1.7615942 ]]


