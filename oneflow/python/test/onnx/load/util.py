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
from collections import OrderedDict
import tempfile
import os
import shutil

import numpy as np
import onnxruntime as ort
import onnx
import torch

import oneflow as flow
import oneflow.typing as tp


def load_pytorch_module_and_check(
    test_case,
    pt_module_class,
    input_size=None,
    input_min_val=-10,
    input_max_val=10,
    train_flag=True,
):
    if input_size is None:
        input_size = (2, 4, 3, 5)
    pt_module = pt_module_class()

    model_weight_save_dir = "/home/zhangxiaoyu/tmp"

    if train_flag == True:

        @flow.global_function(type="train")
        def job_train(x: tp.Numpy.Placeholder(input_size)) -> tp.Numpy:
            x += flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )

            y = flow.from_pytorch(
                pt_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0])
            flow.optimizer.SGD(lr_scheduler).minimize(y)
            return y

    else:

        @flow.global_function(type="predict")
        def job_eval(x: tp.Numpy.Placeholder(input_size)) -> tp.Numpy:
            x += flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )

            y = flow.from_pytorch(
                pt_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )
            return y

    flow.load_variables(flow.checkpoint.get(model_weight_save_dir))

    pt_module = pt_module.to("cuda")
    if train_flag == False:
        pt_module.eval()

    ipt1 = np.random.uniform(
        low=input_min_val, high=input_max_val, size=input_size
    ).astype(np.float32)
    if train_flag == True:
        flow_res = job_train(ipt1)
    else:
        flow_res = job_eval(ipt1)
    pytorch_res = pt_module(torch.tensor(ipt1).to("cuda")).cpu().detach().numpy()
    print(flow_res)
    print("-------------")
    print(pytorch_res)
    np.save("/home/zhangxiaoyu/tmp/flow_res.npy", flow_res)
    np.save("/home/zhangxiaoyu/tmp/pytorch_res.npy", pytorch_res)

    a, b = flow_res.flatten(), pytorch_res.flatten()

    max_idx = np.argmax(np.abs(a - b) / a)
    print("max rel diff is {} at index {}".format(np.max(np.abs(a - b) / a), max_idx))
    print("a[{}]={}, b[{}]={}".format(max_idx, a[max_idx], max_idx, b[max_idx]))
    msg = "success"
    test_case.assertTrue(np.allclose(flow_res, pytorch_res, rtol=1e-4, atol=1e-5), msg)
