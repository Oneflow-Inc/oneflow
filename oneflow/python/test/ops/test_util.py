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
import itertools
import os
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import oneflow as flow
import oneflow.typing as oft

import test_global_storage


def GenCartesianProduct(sets):
    assert isinstance(sets, Iterable)
    for set in sets:
        assert isinstance(set, Iterable)
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            if "gpu" in set:
                set.remove("gpu")
    return itertools.product(*sets)


def GenArgList(arg_dict):
    assert isinstance(arg_dict, OrderedDict)
    assert all([isinstance(x, list) for x in arg_dict.values()])
    sets = [arg_set for _, arg_set in arg_dict.items()]
    return GenCartesianProduct(sets)


def GenArgDict(arg_dict):
    return [dict(zip(arg_dict.keys(), x)) for x in GenArgList(arg_dict)]


class Args:
    def __init__(self, flow_args, tf_args=None):
        super().__init__()
        if tf_args is None:
            tf_args = flow_args
        self.flow_args = flow_args
        self.tf_args = tf_args

    def __str__(self):
        return "flow_args={} tf_args={}".format(self.flow_args, self.tf_args)

    def __repr__(self):
        return self.__str__()


def RunOneflowOp(device_type, flow_op, x, flow_args):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def FlowJob(x: oft.Numpy.Placeholder(x.shape)):
        with flow.scope.placement(device_type, "0:0"):
            x += flow.get_variable(
                name="v1",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            loss = flow_op(x, *flow_args)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0]), momentum=0
            ).minimize(loss)

            flow.watch_diff(x, test_global_storage.Setter("x_diff"))

            return loss

    # OneFlow
    y = FlowJob(x).get().numpy()
    x_diff = test_global_storage.Get("x_diff")
    return y, x_diff


def RunTensorFlowOp(tf_op, x, tf_args):
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        y = tf_op(x, *tf_args)
    x_diff = tape.gradient(y, x)
    return y.numpy(), x_diff.numpy()


def CompareOpWithTensorFlow(
    device_type,
    flow_op,
    tf_op,
    input_shape,
    op_args=None,
    input_minval=-10,
    input_maxval=10,
    y_rtol=1e-5,
    y_atol=1e-5,
    x_diff_rtol=1e-5,
    x_diff_atol=1e-5,
):
    assert device_type in ["gpu", "cpu"]
    if op_args is None:
        flow_args, tf_args = [], []
    else:
        flow_args, tf_args = op_args.flow_args, op_args.tf_args

    x = np.random.uniform(low=input_minval, high=input_maxval, size=input_shape).astype(
        np.float32
    )
    of_y, of_x_diff, = RunOneflowOp(device_type, flow_op, x, flow_args)
    tf_y, tf_x_diff = RunTensorFlowOp(tf_op, x, tf_args)

    assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)
    assert np.allclose(of_x_diff, tf_x_diff, rtol=x_diff_rtol, atol=x_diff_atol)


type_name_to_flow_type = {
    "float16": flow.float16,
    "float32": flow.float32,
    "double": flow.double,
    "int8": flow.int8,
    "int32": flow.int32,
    "int64": flow.int64,
    "char": flow.char,
    "uint8": flow.uint8,
}

type_name_to_np_type = {
    "float16": np.float16,
    "float32": np.float32,
    "double": np.float64,
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
    "char": np.byte,
    "uint8": np.uint8,
}
