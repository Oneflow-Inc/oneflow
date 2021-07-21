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
# import torch
import numpy as np

import oneflow.experimental as flow

# x = np.random.randn(2)
# y = np.random.randn(2)

# print(x, y)

# out  = np.dot(x, y)

# x = torch.tensor([2.3, 2.6], requires_grad=True)
# y = torch.tensor([3.7, 8.6], requires_grad=True)


# z = torch.dot(x, y)

# z.backward()
# print(x.grad)
# print(y.grad)


dx, dy = flow.F.dot_grad(
    flow.tensor([1.2, 2.3]), flow.tensor([3.0, 4.0]), flow.tensor([1.0])
)

print(dx, dy)


a = flow.tensor([8.6, 3.7,], requires_grad=True)
b = flow.tensor([3.7, 8.6], requires_grad=True)

c = flow.dot(a, b)
c.backward()

print(a.grad)
print(b.grad)

e = flow.tensor([8.6, 3.7], requires_grad=True, device=flow.device("cuda"))
f = flow.tensor([3.7, 8.6], requires_grad=True, device=flow.device("cuda"))

g = flow.dot(e, f)
g.backward()

print(e.grad)
print(f.grad)
