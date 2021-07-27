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

import typing
import unittest
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import test_global_storage
from test_util import GenArgDict, GenArgList, type_name_to_flow_type

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(
    device_type,
    a_shape,
    b_shape,
    transpose_a,
    transpose_b,
    data_type,
    fuse_add_to_output,
    enable_tf32,
    alpha,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.enable_fuse_add_to_output(fuse_add_to_output)
    flow.config.enable_tensor_float_32_compute(enable_tf32)
    if data_type == "float16":
        dtype = flow.float
    else:
        dtype = type_name_to_flow_type[data_type]

    @flow.global_function(type="train", function_config=func_config)
    def MatmulJob():
        with flow.scope.placement(device_type, "0:0"):
            a = flow.get_variable(
                "a",
                shape=a_shape,
                dtype=dtype,
                initializer=flow.random_uniform_initializer(minval=0, maxval=1),
                trainable=True,
            )
            b = flow.get_variable(
                "b",
                shape=b_shape,
                dtype=dtype,
                initializer=flow.random_uniform_initializer(minval=0, maxval=1),
                trainable=True,
            )
            if data_type == "float16":
                out = flow.matmul(
                    flow.cast(a, dtype=flow.float16),
                    flow.cast(b, dtype=flow.float16),
                    transpose_a,
                    transpose_b,
                    alpha,
                )
                c = flow.get_variable(
                    "c",
                    shape=out.shape,
                    dtype=dtype,
                    initializer=flow.random_uniform_initializer(minval=-1, maxval=1),
                    trainable=True,
                )
                loss = flow.cast(
                    out + flow.cast(c, dtype=flow.float16), dtype=flow.float
                )
            else:
                out = flow.matmul(a, b, transpose_a, transpose_b, alpha)
                c = flow.get_variable(
                    "c",
                    shape=out.shape,
                    dtype=dtype,
                    initializer=flow.random_uniform_initializer(minval=-1, maxval=1),
                    trainable=True,
                )
                loss = out + c
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0.0001]), momentum=0
            ).minimize(loss)
            flow.watch(a, test_global_storage.Setter("a"))
            flow.watch_diff(a, test_global_storage.Setter("a_diff"))
            flow.watch(b, test_global_storage.Setter("b"))
            flow.watch_diff(b, test_global_storage.Setter("b_diff"))
            flow.watch(c, test_global_storage.Setter("c"))
            flow.watch_diff(c, test_global_storage.Setter("c_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))
            return loss

    of_out = MatmulJob().get()
    with tf.GradientTape(persistent=True) as tape:
        a = tf.Variable(test_global_storage.Get("a"))
        b = tf.Variable(test_global_storage.Get("b"))
        c = tf.Variable(test_global_storage.Get("c"))
        if data_type == "float16":
            a = tf.cast(a, tf.float16)
            b = tf.cast(b, tf.float16)
            c = tf.cast(c, tf.float16)
        tf_out = tf.matmul(a, b, transpose_a, transpose_b)
        tf_out = tf_out * alpha
        tf_out = tf_out + c
        if data_type == "float16":
            tf_out = tf.cast(tf_out, tf.float32)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_a_diff = tape.gradient(tf_out, a, loss_diff)
    tf_b_diff = tape.gradient(tf_out, b, loss_diff)
    tf_c_diff = tape.gradient(tf_out, c, loss_diff)
    if data_type == "float16":
        tolerance = 0.002
    else:
        tolerance = 0.001
    assert np.allclose(
        of_out.numpy(), tf_out.numpy(), rtol=tolerance, atol=tolerance
    ), np.max(np.abs(of_out.numpy() - tf_out.numpy()))
    assert np.allclose(
        test_global_storage.Get("a_diff"),
        tf_a_diff.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )
    assert np.allclose(
        test_global_storage.Get("b_diff"),
        tf_b_diff.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )
    assert np.allclose(
        test_global_storage.Get("c_diff"),
        tf_c_diff.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


def filter_args(arg_list):
    def trans_shape(shape):
        tmp_shape = shape[:-2]
        tmp_shape += (shape[-1], shape[-2])
        return tmp_shape

    ret = []
    for arg in arg_list:
        a_shape = arg[1]
        b_shape = arg[2]
        if arg[3]:
            a_shape = trans_shape(a_shape)
        if arg[4]:
            b_shape = trans_shape(b_shape)
        if a_shape[-1] == b_shape[-2]:
            ret.append(tuple(arg))
    return ret


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["a_shape"] = [(512, 256), (256, 512)]
    arg_dict["b_shape"] = [(256, 1024), (1024, 256)]
    arg_dict["transpose_a"] = [True, False]
    arg_dict["transpose_b"] = [True, False]
    arg_dict["data_type"] = ["float16", "float32", "double"]
    arg_dict["fuse_add_to_output"] = [True, False]
    arg_dict["enable_tf32"] = [True, False]
    arg_dict["alpha"] = [1.5, 1]
    matmul_args = filter_args(GenArgList(arg_dict))
    arg_dict.clear()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["a_shape"] = [(10, 10, 64, 32), (10, 10, 32, 64)]
    arg_dict["b_shape"] = [(10, 10, 32, 128), (10, 10, 128, 32)]
    arg_dict["transpose_a"] = [True, False]
    arg_dict["transpose_b"] = [True, False]
    arg_dict["data_type"] = ["float16", "float32", "double"]
    arg_dict["fuse_add_to_output"] = [True, False]
    arg_dict["enable_tf32"] = [True, False]
    arg_dict["alpha"] = [2.0]
    batch_matmul_args = filter_args(GenArgList(arg_dict))
    return matmul_args + batch_matmul_args


def filter_args_v2(arg_dict_list):
    def trans_shape(shape):
        tmp_shape = shape[:-2]
        tmp_shape += (shape[-1], shape[-2])
        return tmp_shape

    ret = []
    for arg_dict in arg_dict_list:
        if arg_dict["transpose_a"]:
            a_shape = trans_shape(arg_dict["a_shape"])
        else:
            a_shape = arg_dict["a_shape"]
        if arg_dict["transpose_b"]:
            b_shape = trans_shape(arg_dict["b_shape"])
        else:
            b_shape = arg_dict["b_shape"]
        if a_shape[-1] != b_shape[-2]:
            continue
        if arg_dict["device_type"] == "cpu" and (
            arg_dict["data_type"] == "float16" or arg_dict["enable_tf32"] is True
        ):
            continue
        if arg_dict["data_type"] != "float32" and arg_dict["enable_tf32"] is True:
            continue
        if (
            arg_dict["test_add_to_output"] is False
            and arg_dict["fuse_add_to_output"] is True
        ):
            continue
        arg_dict["atol"] = 1e-05
        ret.append(arg_dict)
    return ret


def gen_args():
    args = OrderedDict()
    args["a_shape"] = [(10, 3, 4), (7, 6, 8)]
    args["b_shape"] = [(4, 5), (10, 8)]
    args["transpose_a"] = [False]
    args["transpose_b"] = [True, False]
    args["alpha"] = [1.5, 1]
    args["data_type"] = ["float16", "float32", "double"]
    args["device_type"] = ["gpu", "cpu"]
    args["test_add_to_output"] = [True, False]
    args["fuse_add_to_output"] = [True, False]
    args["enable_tf32"] = [True, False]
    return filter_args_v2(GenArgDict(args))


def get_lr_scheduler():
    return flow.optimizer.PiecewiseConstantScheduler([], [0.0001])


def get_optimizer():
    return flow.optimizer.SGD(get_lr_scheduler(), momentum=0)


def make_matmul_func(
    a_shape,
    b_shape,
    trans_a,
    trans_b,
    alpha,
    dtype,
    device_type,
    test_add_to_output,
    fuse_add_to_output,
    tf32,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    flow.config.enable_tensor_float_32_compute(tf32)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.enable_fuse_add_to_output(fuse_add_to_output)
    func_config.default_placement_scope(flow.scope.placement(device_type, "0:0"))

    @flow.global_function(type="train", function_config=func_config)
    def matmul_job() -> typing.Tuple[
        flow.typing.Numpy, flow.typing.Numpy, flow.typing.Numpy, flow.typing.Numpy
    ]:
        a_var = flow.get_variable(
            "a",
            shape=a_shape,
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(minval=0, maxval=1),
            trainable=True,
        )
        b_var = flow.get_variable(
            "b",
            shape=b_shape,
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(minval=0, maxval=1),
            trainable=True,
        )
        flow.watch_diff(a_var, test_global_storage.Setter("a_diff"))
        flow.watch_diff(b_var, test_global_storage.Setter("b_diff"))
        if dtype is flow.float16:
            a = flow.amp_white_identity(a_var)
            b = flow.amp_white_identity(b_var)
        else:
            a = a_var
            b = b_var
        c = flow.matmul(a, b, trans_a, trans_b, alpha)
        add_to = flow.get_variable(
            "c",
            shape=c.shape,
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(minval=-1, maxval=1),
            trainable=True,
        )
        if test_add_to_output:
            flow.watch_diff(add_to, test_global_storage.Setter("add_to_diff"))
            if dtype is flow.float16:
                add_to = flow.amp_white_identity(add_to)
            c = c + add_to
        flow.watch_diff(c, test_global_storage.Setter("c_diff"))
        get_optimizer().minimize(c)
        return (a_var, b_var, add_to, c)

    return matmul_job


def np_matmul(a, b, trans_a=False, trans_b=False, bias=None, alpha=None):
    assert len(a.shape) >= 2
    assert len(b.shape) >= 2

    def transpose(x):
        if len(x.shape) == 2:
            x = np.transpose(x)
        else:
            perm = list(range(x.ndim)[:-2]) + [x.ndim - 1, x.ndim - 2]
            x = np.transpose(x, perm)
        return x

    if trans_a:
        a = transpose(a)
    if trans_b:
        b = transpose(b)
    c = np.matmul(a, b)
    if alpha is not None:
        c = c * float(alpha)
    if bias is not None:
        c = c + bias
    return c


def compare_with_np(
    test_case,
    a_shape,
    b_shape,
    transpose_a,
    transpose_b,
    alpha=1.0,
    data_type="float32",
    device_type="gpu",
    test_add_to_output=False,
    fuse_add_to_output=False,
    enable_tf32=False,
    rtol=1e-05,
    atol=1e-08,
):
    def print_dbg_info(lhs=None, rhs=None):
        print(
            f"a_shape: {a_shape}, b_shape: {b_shape}, transpose_a: {transpose_a}, transpose_b: {transpose_b}, alpha: {alpha}, data_type: {data_type}, device_type: {device_type}, test_add_to_output: {test_add_to_output}, fuse_add_to_output: {fuse_add_to_output}, enable_tf32: {enable_tf32}"
        )
        if lhs is not None:
            print(f"lhs: {lhs.shape}\n{lhs}")
        if rhs is not None:
            print(f"rhs: {rhs.shape}\n{rhs}")
        if lhs is not None and rhs is not None:
            diff = lhs - rhs
            print(f"abs diff mean: {np.abs(diff).mean()}")
            print(f"abs diff max: {np.abs(diff).max()}")

    dtype = type_name_to_flow_type[data_type]
    matmul_fn = make_matmul_func(
        a_shape,
        b_shape,
        transpose_a,
        transpose_b,
        alpha,
        dtype,
        device_type,
        test_add_to_output,
        fuse_add_to_output,
        enable_tf32,
    )
    (a, b, add_to_output, c) = matmul_fn()
    if test_add_to_output is False:
        add_to_output = None
    c_ = np_matmul(a, b, transpose_a, transpose_b, bias=add_to_output, alpha=alpha)
    comp_c_result = np.allclose(c, c_, rtol, atol)
    if not comp_c_result:
        print_dbg_info(c, c_)
    test_case.assertTrue(comp_c_result)
    c_diff = test_global_storage.Get("c_diff")
    a_diff = test_global_storage.Get("a_diff")
    b_diff = test_global_storage.Get("b_diff")
    if transpose_a:
        raise NotImplementedError
    else:
        a_diff_ = np_matmul(
            c_diff, b, transpose_a, not transpose_b, bias=None, alpha=alpha
        )
    comp_a_diff_result = np.allclose(a_diff, a_diff_, rtol, atol)
    if not comp_a_diff_result:
        print_dbg_info(a_diff, a_diff_)
    test_case.assertTrue(comp_a_diff_result)
    if transpose_b:
        b_diff_ = np_matmul(
            c_diff.reshape((-1, c_diff.shape[-1])),
            a.reshape((-1, a.shape[-1])),
            True,
            transpose_a,
            bias=None,
            alpha=alpha,
        )
    else:
        b_diff_ = np_matmul(
            a.reshape((-1, a.shape[-1])),
            c_diff.reshape((-1, c_diff.shape[-1])),
            not transpose_a,
            False,
            bias=None,
            alpha=alpha,
        )
    comp_b_diff_result = np.allclose(b_diff, b_diff_, rtol, atol)
    if not comp_b_diff_result:
        print_dbg_info(b_diff, b_diff_)
    test_case.assertTrue(comp_b_diff_result)
    if test_add_to_output:
        add_to_diff = test_global_storage.Get("add_to_diff")
        test_case.assertTrue(np.allclose(add_to_diff, c_diff))


@flow.unittest.skip_unless_1n1d()
class TestMatmul(flow.unittest.TestCase):
    def test_matmul(test_case):
        for arg in gen_arg_list():
            if arg[0] == "cpu" and (arg[5] == "float16" or arg[7] == True):
                continue
            if arg[5] != "float32" and arg[7] == True:
                continue
            compare_with_tensorflow(*arg)

    def test_broadcast_matmul(self):
        for arg in gen_args():
            compare_with_np(self, **arg)


if __name__ == "__main__":
    unittest.main()
