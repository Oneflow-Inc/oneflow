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

import os
import unittest
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def fused_cast_scale(x, scale_by_tensor, scale, name):
    return (
        flow.user_op_builder(name)
        .Op("fused_cast_scale")
        .Input("x", [x])
        .Input("scale_by_tensor", [scale_by_tensor])
        .Output("y")
        .Attr("scale", float(scale))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def compare_with_tensorflow(
    device_type,
    input_shape,
    in_dtype,
    out_dtype,
    test_fuse_cast_scale_pass,
    has_scalar_mul,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.enable_fuse_cast_scale(True)

    @flow.global_function(type="predict", function_config=func_config)
    def FusedCastScaleJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "in",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(),
                trainable=True,
            )
            scale = flow.get_variable(
                "scale",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(),
                trainable=False,
            )
            loss = flow.cast(x, dtype=type_name_to_flow_type[in_dtype])
            if test_fuse_cast_scale_pass:
                loss = flow.cast(loss, dtype=type_name_to_flow_type[out_dtype])
                if has_scalar_mul:
                    loss = loss * 0.125
                loss = loss * flow.cast(scale, dtype=type_name_to_flow_type[out_dtype])
            else:
                if has_scalar_mul:
                    scale_val = 0.125
                else:
                    scale_val = 1.0
                loss = fused_cast_scale(
                    loss,
                    flow.cast(scale, dtype=type_name_to_flow_type[out_dtype]),
                    scale=scale_val,
                    name="fused_cast_scale",
                )
            loss = flow.cast(loss, dtype=flow.float)
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch(scale, test_global_storage.Setter("scale"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            return loss

    of_out = FusedCastScaleJob().get()
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        scale = tf.Variable(test_global_storage.Get("scale"))
        tf_out = tf.cast(x, dtype=type_name_to_np_type[in_dtype])
        tf_out = tf.cast(tf_out, dtype=type_name_to_np_type[out_dtype]) * tf.cast(
            scale, dtype=type_name_to_np_type[out_dtype]
        )
        if has_scalar_mul:
            tf_out = tf_out * 0.125
        tf_out = tf.cast(tf_out, dtype=tf.float32)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=1e-05, atol=1e-05)


@flow.unittest.skip_unless_1n1d()
class TestFusedCastScale(flow.unittest.TestCase):
    def test_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(5, 4, 3)]
        arg_dict["in_dtype"] = ["float16", "float32", "double"]
        arg_dict["out_dtype"] = ["float16", "float32", "double"]
        arg_dict["test_fuse_cast_scale_pass"] = [True, False]
        arg_dict["has_scalar_mul"] = [True, False]
        for arg in GenArgList(arg_dict):
            if arg[2] == arg[3]:
                continue
            if arg[4] == True and (arg[2] != "float16" or arg[3] != "float32"):
                continue
            if arg[0] == "cpu" and (arg[2] == "float16" or arg[3] == "float16"):
                continue
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
