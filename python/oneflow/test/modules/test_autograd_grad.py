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


def calc_grad(x):
    #y = x * x * x
    y = x * x * x

    x_grad = flow.autograd.grad(
        outputs=y,
        inputs=x,
        out_grads=flow.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
    print(x_grad)
    return x_grad


x = flow.tensor(10.0, requires_grad=True)
x_grad = calc_grad(x)

x_grad_grad = flow.autograd.grad(
    outputs=x_grad, inputs=x, out_grads=flow.ones_like(x_grad)
)[0]

print(x_grad_grad)
