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

import tempfile
import unittest
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgDict

import oneflow as flow
from oneflow.nn.parameter import Parameter

def compare_with_sgd_foreach(
    test_case,
    device,
    x_shape,
    tensor_num,
    weight_decay,
    learning_rate,
    train_iters,
    reload_state_step,
    save_load_by_pickle,
):
    random_grad_seq = []
    init_value_seq = []

    for _ in range(train_iters):
        random_grad_seq_per_iter = []
        for i in range(tensor_num):
            random_grad_seq_per_iter.append(
                np.random.uniform(size=x_shape).astype(np.float32)
            )
        random_grad_seq.append(random_grad_seq_per_iter)

    for i in range(tensor_num):
        init_value_seq.append(np.random.uniform(size=x_shape).astype(np.float32))


    def _train_with_sgd(multi_tensor):
        x = []
        for value in init_value_seq:
            x.append(Parameter(flow.Tensor(value, device=flow.device(device))))
        sgd = flow.optim.SGD(
            [{"params": x, "lr": learning_rate, "weight_decay": weight_decay,}],
            multi_tensor=multi_tensor
        )

        def train_one_iter(grad):
            loss = 0.0
            for i in range(tensor_num):
                grad_tensor = flow.tensor(
                    grad,
                    dtype=flow.float32,
                    requires_grad=False,
                    device=flow.device(device),
                )
                loss += flow.sum(x[i] * grad_tensor)
            loss.backward()
            sgd.step()
            sgd.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = sgd.state_dict()
                sgd = flow.optim.SGD(x)
                if save_load_by_pickle:
                    with tempfile.TemporaryDirectory() as save_dir:
                        flow.save(state_dict, save_dir)
                        state_dict = flow.load(save_dir)
                sgd.load_state_dict(state_dict)
        flow.cuda.synchronize()

        return x

    a = _train_with_sgd(False)
    b = _train_with_sgd(True)
 
    for i in range(tensor_num):
        test_case.assertTrue(
            np.allclose(
                a[i].numpy().flatten(), b[i].numpy().flatten(), rtol=0.0001, atol=0.0001
            )
        )
    
@flow.unittest.skip_unless_1n1d()
class TestOptimizers(flow.unittest.TestCase):
    def test_sgd_foreach(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["tensor_num"] = [10, 20] 
        arg_dict["weight_decay"] = [0.9, 0.1]
        arg_dict["learning_rate"] = [1.0, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["reload_state_step"] = [5]
        arg_dict["save_load_by_pickle"] = [False, True]
        for arg in GenArgDict(arg_dict):
            compare_with_sgd_foreach(test_case, **arg)

if __name__ == "__main__":
    unittest.main()
