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
import numpy as np
import oneflow as flow

batch_size = 10
embedding_size = 128
self_interaction = False
output_padding = 1
if self_interaction:
    offset = 1
else:
    offset = 0
x_np = np.random.rand(batch_size, embedding_size).astype(np.float32)
ly_np = np.random.rand(batch_size, 26, embedding_size).astype(np.float32)
x = flow.tensor(x_np, device="cuda", requires_grad=True)
ly = flow.tensor(ly_np, device="cuda", requires_grad=True)
li = flow.tensor([i for i in range(27) for j in range(i + offset)])
lj = flow.tensor([j for i in range(27) for j in range(i + offset)])
T = flow.cat([flow.reshape(x, (batch_size, 1, embedding_size)), ly], dim=1)
Z = flow.matmul(T, T, transpose_b=True)
Zflat = Z[:, li, lj]
R = flow.cat([x, Zflat], dim=1)
loss1 = R.sum()
loss1.backward()

dense = flow.tensor(x_np, device="cuda", requires_grad=True)
sparse = flow.tensor(ly_np, device="cuda", requires_grad=True)
y = flow._C.fused_dot_feature_interaction(
    [dense.reshape(batch_size, 1, embedding_size), sparse],
    output_concat=dense,
    self_interaction=self_interaction,
    output_padding=output_padding,
)
print(y.shape)
loss = y.sum()
loss.backward()
print("x grad", np.allclose(x.grad.numpy(), dense.grad.numpy(), rtol=1e-3, atol=1e-4))
print(
    "sparse grad",
    np.allclose(ly.grad.numpy(), sparse.grad.numpy(), rtol=1e-3, atol=1e-4),
)
print(
    "y ", np.allclose(y.numpy()[:, :-output_padding], R.numpy(), rtol=1e-4, atol=1e-4)
)
