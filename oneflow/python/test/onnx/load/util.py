from collections import OrderedDict
import tempfile
import os
import shutil

import numpy as np
import onnxruntime as ort
import onnx
import torch

import oneflow as flow


def load_pytorch_module_and_check(test_case, pt_module_class, input_size=None):
    if input_size is None:
        input_size = (2, 4, 3, 5)
    pt_module = pt_module_class()

    model_weight_save_dir = '/tmp/tmp'
    @flow.global_function(type='train')
    def job(x=flow.FixedTensorDef(input_size)):
        x += flow.get_variable(
            name="trick",
            shape=(1,),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )

        y = flow.from_pytorch(pt_module, x, model_weight_dir=model_weight_save_dir)
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0])
        flow.optimizer.SGD(lr_scheduler).minimize(y)
        return y


    checkpoint = flow.train.CheckPoint()
    checkpoint.load(model_weight_save_dir)

    pt_module = pt_module.to("cuda")
    ipt1 = np.random.uniform(low=-1000, high=1000, size=input_size).astype(np.float32)
    # flow_res = temp_job(ipt1).get().ndarray()
    flow_res = job(ipt1).get().ndarray()
    pytorch_res = pt_module(torch.tensor(ipt1).to("cuda")).cpu().detach().numpy()

    a, b = flow_res.flatten(), pytorch_res.flatten()

    max_idx = np.argmax(np.abs(a - b) / a)
    print("max rel diff is {} at index {}".format(np.max(np.abs(a - b) / a), max_idx))
    print("a[{}]={}, b[{}]={}".format(max_idx, a[max_idx], max_idx, b[max_idx]))
    test_case.assertTrue(np.allclose(flow_res, pytorch_res, rtol=1e-4, atol=1e-5))
