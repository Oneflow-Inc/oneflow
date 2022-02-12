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
import tempfile

import numpy as np
from test_util import GenArgDict
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter

from oneflow.test_utils.automated_test_util import *


def compare_with_numpy_sgd(
    test_case,
    placement,
    sbp,
    device,
    x_shape,
    momentum,
    weight_decay,
    learning_rate,
    train_iters,
    reload_state_step,
    save_load_by_pickle,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    def train_by_oneflow():
        x = flow.Tensor(init_value, device=flow.device(device))
        x = x.to_consistent(placement,sbp)
        x = Parameter(x)
        sgd = flow.optim.SGD(
            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                }
            ]
        )

        def train_one_iter(grad):
            grad_tensor = flow.tensor(
                grad,
                dtype=flow.float32,
                requires_grad=False,
                device=flow.device(device),
            )
            grad_tensor = grad_tensor.to_consistent(placement,sbp)
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            sgd.step()
            sgd.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            # test state_dict/load_state_dict
            if i == reload_state_step:
                state_dict = sgd.state_dict()
                sgd = flow.optim.SGD([x])
                if save_load_by_pickle:
                    with tempfile.TemporaryDirectory() as save_dir:
                        flow.save(state_dict, save_dir)
                        state_dict = flow.load(save_dir)
                sgd.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)

        def train_one_iter(grad):
            grad = grad + weight_decay * x
            v = momentum * vt - learning_rate * grad
            param = x + v
            return (param, v)

        for i in range(train_iters):
            (x, vt) = train_one_iter(random_grad_seq[i])
        return x
    

    oneflow_res = train_by_oneflow()
    numpy_res = train_by_numpy()

    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.allclose(
                oneflow_res.to_local().numpy().flatten(), numpy_res.flatten(), rtol=0.0001, atol=0.0001
            )
        )



class TestOptimizers(flow.unittest.TestCase):
    @consistent
    def test_sgd(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(8,8)]
        arg_dict["momentum"] = [0.0, 0.9]
        arg_dict["weight_decay"] = [0.0, 0.9]
        arg_dict["learning_rate"] = [1, 0.1]
        arg_dict["train_iters"] = [10]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        for arg in GenArgDict(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=2):
                    compare_with_numpy_sgd(test_case, placement, sbp, **arg)



if __name__ == "__main__":
    unittest.main()

#单卡出错 :  can't find available sbp signature
