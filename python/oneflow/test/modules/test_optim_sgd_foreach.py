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
import time
import pdb

import numpy as np
from oneflow import optim
from oneflow.test_utils.test_util import GenArgDict
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter
import oneflow.profiler as profiler
from oneflow.profiler.events import CustomEvent, KernelEvent

def compare_with_sgd_foreach(
    test_case,
    device,
    x_shape,
    tensor_num,
    # momentum,
    # dampening,
    # nesterov,
    # maximize,
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

    def get_run_time(func):
        def wrap(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            
            print("func:{}\t device:{}\t shape:[{}, {}]\t time:{}".format(
                func.__name__, device, x_shape, tensor_num, end-start))
            return result
        return wrap

    def _train_with_sgd(foreach):
        x = []
        for value in init_value_seq:
            x.append(Parameter(flow.Tensor(value, device=flow.device(device))))
        sgd = flow.optim.SGD(
            [{"params": x, "lr": learning_rate, "weight_decay": weight_decay,}],
            # momentum=momentum,
            # dampening=dampening,
            # nesterov=nesterov,
            # maximize=maximize,
            foreach=foreach
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
        return x

    def get_event(events, name: str, input_shapes: str = "-"):
        for item in events:
            if isinstance(item, CustomEvent):
                print(item.name)
                if item.name == name:
                    return item
            if isinstance(item, KernelEvent):
                print(item.name, item.input_shapes)
                if item.name == name and item.input_shapes == input_shapes:
                    return item
        return None

    def train_with_profiler(foreach):
        activities = [profiler.ProfilerActivity.CPU]
        if device == 'cuda':
            activities.append(profiler.ProfilerActivity.CUDA)

        with profiler.profile(
            activities=activities,
            record_shapes=True,
            record_bandwidth_for_cuda=True,
        ) as prof:
            with profiler.record_function("foreach:{}".format(foreach)) as f:
                ret =  _train_with_sgd(foreach)

        events = prof.key_averages(group_by_input_shape=True)

        optim_event = get_event(
            events, "conv2d", "[(2,3,32,32), (6,3,5,5)]"
        )
        return ret
        test_case.assertIsNotNone(optim_event)
        print('cpu time:{}, cpu totol time:{}, cuda time:{}, cuda total time:{}, band width:{}'.format(
            optim_event.cpu_time, optim_event.cpu_time_total, optim_event.cuda_time,
            optim_event.cuda_time_total, optim_event.bandwidth
        ))
        print('-' * 100)

        return ret

    def train_not_foreach():
        return train_with_profiler(False)

    def train_foreach():
        return train_with_profiler(True)

    a = train_not_foreach()
    b = train_foreach()
 
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
        #arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["device"] = ["cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["tensor_num"] = [10]
        # arg_dict["momentum"] = [0.0, 0.9]
        # arg_dict["dampening"] = [0.0, 0.9]
        # arg_dict["nesterov"] = [True, False]
        # arg_dict["maximize"] = [True, False]
        arg_dict["weight_decay"] = [0.9]
        arg_dict["learning_rate"] = [0.1]
        arg_dict["train_iters"] = [10, 100]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [True]
        for arg in GenArgDict(arg_dict):
            compare_with_sgd_foreach(test_case, **arg)

if __name__ == "__main__":
    unittest.main()