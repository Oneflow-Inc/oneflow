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
import onnxruntime as ort
import onnx
from collections import OrderedDict
import tempfile
import os
import shutil


def convert_to_onnx_and_check(
    job_func,
    print_rel_diff=False,
    explicit_init=True,
    external_data=False,
    ort_optimize=True,
    opset=None,
):
    check_point = flow.train.CheckPoint()
    if explicit_init:
        # it is a trick to keep check_point.save() from hanging when there is no variable
        @flow.global_function(flow.FunctionConfig())
        def add_var():
            return flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(),
            )

        check_point.init()
    flow_weight_dir = tempfile.TemporaryDirectory()
    check_point.save(flow_weight_dir.name)
    # TODO(daquexian): a more elegant way?
    while not os.path.exists(os.path.join(flow_weight_dir.name, "snapshot_done")):
        pass
    onnx_model_dir = tempfile.TemporaryDirectory()
    onnx_model_path = os.path.join(onnx_model_dir.name, "model.onnx")
    flow.onnx.export(
        job_func,
        flow_weight_dir.name,
        onnx_model_path,
        opset=opset,
        external_data=external_data,
    )
    flow_weight_dir.cleanup()
    ort_sess_opt = ort.SessionOptions()
    ort_sess_opt.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        if ort_optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    sess = ort.InferenceSession(onnx_model_path, sess_options=ort_sess_opt)
    onnx_model_dir.cleanup()
    assert len(sess.get_outputs()) == 1
    assert len(sess.get_inputs()) <= 1
    ipt_dict = OrderedDict()
    for ipt in sess.get_inputs():
        ipt_data = np.random.uniform(low=-10, high=10, size=ipt.shape).astype(
            np.float32
        )
        ipt_dict[ipt.name] = ipt_data

    onnx_res = sess.run([], ipt_dict)[0]
    oneflow_res = job_func(*ipt_dict.values()).get().numpy()
    if print_rel_diff:
        a = onnx_res.flatten()
        b = oneflow_res.flatten()
        max_idx = np.argmax(np.abs(a - b) / a)
        print(
            "max rel diff is {} at index {}".format(np.max(np.abs(a - b) / a), max_idx)
        )
        print("a[{}]={}, b[{}]={}".format(max_idx, a[max_idx], max_idx, b[max_idx]))
    assert np.allclose(onnx_res, oneflow_res, rtol=1e-4, atol=1e-5)
