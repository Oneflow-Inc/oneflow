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
import os
import random as random_util

import numpy as np
from oneflow.test_utils.test_util import GenArgDict
from oneflow.test_utils.automated_test_util import random_bool, random_device
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter


def compare_with_numpy_sgd(
    test_case,
    device,
    x_shape,
    momentum,
    dampening,
    nesterov,
    maximize,
    weight_decay,
    learning_rate,
    train_iters,
    reload_state_step,
    save_load_by_pickle,
    contiguous_params,
    fused,
    tensor_num,
):
    random_grad_seq = []
    init_value_seq = []

    for i in range(tensor_num):
        init_value_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    for _ in range(train_iters):
        random_grad_seq_per_iter = []
        for i in range(tensor_num):
            random_grad_seq_per_iter.append(
                np.random.uniform(size=x_shape).astype(np.float32)
            )
        random_grad_seq.append(random_grad_seq_per_iter)

    def train_by_oneflow():
        x = []
        for i in range(tensor_num):
            x.append(
                Parameter(flow.Tensor(init_value_seq[i], device=flow.device(device)))
            )

        sgd = flow.optim.SGD(
            [{"params": x, "lr": learning_rate, "weight_decay": weight_decay,}],
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            contiguous_params=contiguous_params,
            fused=fused,
        )

        def train_one_iter(grad):
            loss = 0.0
            for i in range(tensor_num):
                grad_tensor = flow.tensor(
                    grad[i],
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
            # test state_dict/load_state_dict
            if i == reload_state_step:
                state_dict = sgd.state_dict()
                sgd = flow.optim.SGD(x, contiguous_params=contiguous_params)
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                sgd.load_state_dict(state_dict)
        return x

    def train_by_numpy(tensor_idx):
        x = init_value_seq[tensor_idx]
        vt = np.zeros_like(x)

        def train_one_iter(grad):
            grad = grad + weight_decay * x
            if momentum > 0.0:
                next_momentum = momentum * vt + (1 - dampening) * grad
                v = next_momentum

                if nesterov:
                    grad += momentum * next_momentum
                else:
                    grad = next_momentum

                alpha = -learning_rate
                if maximize:
                    alpha = learning_rate
                next_model = x + alpha * grad
                param = next_model
            else:
                v = learning_rate * grad
                param = x - v
            return (param, v)

        for i in range(train_iters):
            (x, vt) = train_one_iter(random_grad_seq[i][tensor_idx])
        return x

    oneflow_res = train_by_oneflow()
    numpy_res = []
    for i in range(tensor_num):
        numpy_res.append(train_by_numpy(i))

    for i in range(tensor_num):
        test_case.assertTrue(
            np.allclose(
                oneflow_res[i].numpy().flatten(),
                numpy_res[i].flatten(),
                rtol=0.0001,
                atol=0.0001,
            )
        )


def compare_with_numpy_sgd_clip_grad(
    test_case,
    device,
    x_shape,
    momentum,
    dampening,
    nesterov,
    maximize,
    weight_decay,
    learning_rate,
    clip_grad_max_norm,
    clip_grad_norm_type,
    train_iters,
    reload_state_step,
    save_load_by_pickle,
    contiguous_params,
    fused,
    tensor_num,
):
    random_grad_seq = []
    init_value_seq = []

    for i in range(tensor_num):
        init_value_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    for _ in range(train_iters):
        random_grad_seq_per_iter = []
        for i in range(tensor_num):
            random_grad_seq_per_iter.append(
                np.random.uniform(size=x_shape).astype(np.float32)
            )
        random_grad_seq.append(random_grad_seq_per_iter)

    def train_by_oneflow():
        x = []
        for i in range(tensor_num):
            x.append(
                Parameter(flow.Tensor(init_value_seq[i], device=flow.device(device)))
            )

        sgd = flow.optim.SGD(
            [
                {
                    "params": x,
                    "lr": learning_rate,
                    "dampening": dampening,
                    "nesterov": nesterov,
                    "maximize": maximize,
                    "weight_decay": weight_decay,
                    "clip_grad_max_norm": clip_grad_max_norm,
                    "clip_grad_norm_type": clip_grad_norm_type,
                }
            ],
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            contiguous_params=contiguous_params,
            fused=fused,
        )

        def train_one_iter(grad):
            loss = 0.0
            for i in range(tensor_num):
                grad_tensor = flow.tensor(
                    grad[i],
                    dtype=flow.float32,
                    requires_grad=False,
                    device=flow.device(device),
                )
                loss += flow.sum(x[i] * grad_tensor)
            loss.backward()
            sgd.clip_grad()
            sgd.step()
            sgd.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            # test state_dict/load_state_dict
            if i == reload_state_step:
                state_dict = sgd.state_dict()
                sgd = flow.optim.SGD(x, contiguous_params=contiguous_params)
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                sgd.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value_seq
        vt = np.zeros_like(x)

        def train_one_iter(grad):
            total_norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )

            for i in range(tensor_num):
                grad[i] = grad[i] + weight_decay * x[i]
                if momentum > 0.0:
                    next_momentum = momentum * vt[i] + (1 - dampening) * grad[i]
                    vt[i] = next_momentum

                    if nesterov:
                        grad[i] += momentum * next_momentum
                    else:
                        grad[i] = next_momentum

                    alpha = -learning_rate
                    if maximize:
                        alpha = learning_rate
                    x[i] = x[i] + alpha * grad[i]
                else:
                    vt[i] = learning_rate * grad[i]
                    x[i] = x[i] - vt[i]

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
        return x

    oneflow_res = train_by_oneflow()
    numpy_res = train_by_numpy()

    for i in range(tensor_num):
        test_case.assertTrue(
            np.allclose(
                oneflow_res[i].numpy().flatten(),
                numpy_res[i].flatten(),
                rtol=0.0001,
                atol=0.0001,
            )
        )


@flow.unittest.skip_unless_1n1d()
class TestOptimizers(flow.unittest.TestCase):
    def test_sgd(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = [random_device().value()]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["momentum"] = [0.0, 0.9]
        arg_dict["dampening"] = [0.0, 0.9]
        arg_dict["nesterov"] = [random_bool().value()]
        arg_dict["maximize"] = [random_bool().value()]
        arg_dict["weight_decay"] = [0.0, 0.9]
        arg_dict["learning_rate"] = [1, 0.1]
        arg_dict["train_iters"] = [10]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [random_bool().value()]
        arg_dict["contiguous_params"] = [random_bool().value()]
        arg_dict["fused"] = [random_bool().value()]
        arg_dict["tensor_num"] = [1, 4]
        for arg in GenArgDict(arg_dict):
            compare_with_numpy_sgd(test_case, **arg)

    def test_sgd_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = [random_device().value()]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["momentum"] = [0.0, 0.9]
        arg_dict["dampening"] = [0.0, 0.9]
        arg_dict["nesterov"] = [random_bool().value()]
        arg_dict["maximize"] = [random_bool().value()]
        arg_dict["weight_decay"] = [0.0, 0.9]
        arg_dict["learning_rate"] = [1, 0.1]
        arg_dict["clip_grad_max_norm"] = [0, 0.5, 1.0]
        arg_dict["clip_grad_norm_type"] = random_util.sample(
            ["inf", "-inf", 0.0, 1.0, 2.0, 3.5], k=3
        )
        arg_dict["train_iters"] = [10]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [random_bool().value()]
        arg_dict["contiguous_params"] = [random_bool().value()]
        arg_dict["fused"] = [random_bool().value()]
        arg_dict["tensor_num"] = [1, 4]
        for arg in GenArgDict(arg_dict):
            compare_with_numpy_sgd_clip_grad(test_case, **arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_eager_global_zero_grad_sbp(test_case):
        x = flow.nn.Parameter(
            flow.zeros((10,)).to_global(
                sbp=flow.sbp.broadcast, placement=flow.placement("cuda", [0])
            )
        )
        x.grad = flow.ones_like(x)
        t = x.grad
        test_case.assertEqual(len(t.sbp), 1)
        test_case.assertEqual(t.sbp[0], flow.sbp.broadcast)
        optimizer = flow.optim.SGD([x])
        optimizer.zero_grad()
        test_case.assertTrue(np.allclose(t.numpy(), 0.0))
        test_case.assertEqual(len(t.sbp), 1)
        test_case.assertEqual(t.sbp[0], flow.sbp.partial_sum)


if __name__ == "__main__":
    unittest.main()
