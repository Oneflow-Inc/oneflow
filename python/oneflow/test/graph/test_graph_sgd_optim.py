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
import unittest
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.unittest

@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
def compare_with_numpy_sgd(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    momentum,
    weight_decay,
    eps,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    class CustomModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.para0 = flow.nn.Parameter(
                flow.Tensor(init_value, device=flow.device(device))
            )

        def forward(self, mask):
            return self.para0 * mask

    simp_module = CustomModule()
    simp_module.to("cuda")
    simp_module.train()

    sgd0 = flow.optim.SGD(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "momentum": momentum,
                "weight_decay": weight_decay,
            }
        ]
    )

    class CustomSGDGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer("sgd", sgd0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    sgd_graph = CustomSGDGraph()
    for i in range(train_iters):
        mask_tensor = flow.Tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        sgd_x = sgd_graph(mask_tensor)
        of_res_list.append(simp_module.para0.numpy())

    np_res_list = []

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)

        def np_train_one_iter(grad):
            grad = grad + weight_decay * x
            v = momentum * vt - learning_rate * grad
            param = x + v
            return (param, v)

        for i in range(train_iters):
            (x, vt) = np_train_one_iter(random_grad_seq[i])
            np_res_list.append(x)

    train_by_numpy()

    test_case.assertTrue(np.allclose(np_res_list, of_res_list, rtol=0.001, atol=0.001))


@flow.unittest.skip_unless_1n1d()
class TestSGD(flow.unittest.TestCase):
    def test_sgd1(test_case):
        compare_with_numpy_sgd(
            test_case,
            device="cuda",
            x_shape=(1,),
            learning_rate=1,
            momentum=0.9,
            train_iters=10,
            weight_decay=0.0,
            eps=1e-8,
        )

    def test_sgd2(test_case):
        compare_with_numpy_sgd(
            test_case,
            device="cuda",
            x_shape=(1,),
            learning_rate=0.01,
            momentum=0.0,
            train_iters=10,
            weight_decay=0.0005,
            eps=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
