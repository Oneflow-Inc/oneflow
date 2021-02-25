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

import numpy as np
import oneflow as flow
from test_util import GenArgList
import oneflow.typing as oft
import os


def _test_split_to_split(
    test_case, src_device_type, dst_device_type, src_axis, dst_axis,
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    def build_s2s(input_blob, src_device_num, dst_device_num):
        with flow.scope.placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(
                input_blob.with_distribute(flow.distribute.split(src_axis))
            )
        with flow.scope.placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.split(dst_axis)))
        return dst

    @flow.global_function(function_config=func_config)
    def split_to_split_job(input_blob: oft.Numpy.Placeholder((96, 96))):
        result_list = []
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                result_list.append(build_s2s(input_blob, i, j))
        return tuple(result_list)

    x = np.random.rand(96, 96).astype(np.float32)
    result_tuple = split_to_split_job(x).get()
    for out in result_tuple:
        test_case.assertTrue(np.array_equal(x, out.numpy()))


def _test_split_to_split_enable_all_to_all(
    test_case, src_device_type, dst_device_type, src_device_num, dst_device_num,
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    flow.config.collective_boxing.nccl_enable_all_to_all(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    def build_s2s_all2all(input_blob, src_axis, dst_axis):
        with flow.scope.placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(
                input_blob.with_distribute(flow.distribute.split(src_axis))
            )
        with flow.scope.placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.split(dst_axis)))
        return dst

    @flow.global_function(function_config=func_config)
    def split_to_split_all2all_job(input_blob: oft.Numpy.Placeholder((32, 16, 64, 48))):
        result_list = []
        for i in (0, 1, 2, 3):
            for j in (0, 1, 2, 3):
                if i == j:
                    continue
                result_list.append(build_s2s_all2all(input_blob, i, j))
        return tuple(result_list)

    x = np.random.rand(32, 16, 64, 48).astype(np.float32)
    result_tuple = split_to_split_all2all_job(x).get()
    for out in result_tuple:
        test_case.assertTrue(np.array_equal(x, out.numpy()))


def _test_split_to_broadcast(
    test_case, src_device_type, dst_device_type, src_axis,
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    def build_s2b(input_blob, src_device_num, dst_device_num):
        with flow.scope.placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(
                input_blob.with_distribute(flow.distribute.split(src_axis))
            )
        with flow.scope.placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.broadcast()))
        return dst

    @flow.global_function(function_config=func_config)
    def split_to_broadcast_job(input_blob: oft.Numpy.Placeholder((96, 96))):
        result_list = []
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                result_list.append(build_s2b(input_blob, i, j))
        return tuple(result_list)

    x = np.random.rand(96, 96).astype(np.float32)
    result_tuple = split_to_broadcast_job(x).get()
    for out in result_tuple:
        test_case.assertTrue(np.array_equal(x, out.numpy()))


def _test_broadcast_to_split(
    test_case, src_device_type, dst_device_type, dst_axis,
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    def build_b2s(input_blob, src_device_num, dst_device_num):
        with flow.scope.placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(input_blob.with_distribute(flow.distribute.broadcast()))
        with flow.scope.placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.split(dst_axis)))
        return dst

    @flow.global_function(function_config=func_config)
    def broadcast_to_split_job(input_blob: oft.Numpy.Placeholder((96, 96))):
        result_list = []
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                result_list.append(build_b2s(input_blob, i, j))
        return tuple(result_list)

    x = np.random.rand(96, 96).astype(np.float32)
    result_tuple = broadcast_to_split_job(x).get()
    for out in result_tuple:
        test_case.assertTrue(np.array_equal(x, out.numpy()))


def _test_partial_sum_to_split(
    test_case, src_device_type, dst_device_type, dst_axis,
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    def build_p2s(input_blob, src_device_num, dst_device_num):
        with flow.scope.placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(input_blob.with_distribute(flow.distribute.split(0)))
            src = flow.math.reduce_sum(src, axis=0)
        with flow.scope.placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.split(dst_axis)))
        return dst

    @flow.global_function(function_config=func_config)
    def partial_sum_to_split_job(input_blob: oft.Numpy.Placeholder((96, 96, 96))):
        result_list = []
        for i in (2, 3):
            for j in (1, 2, 3):
                result_list.append(build_p2s(input_blob, i, j))
        return tuple(result_list)

    x = np.random.uniform(-1e-5, 1e-5, (96, 96, 96)).astype(np.float32)
    result_tuple = partial_sum_to_split_job(x).get()
    for out in result_tuple:
        test_case.assertTrue(np.allclose(np.sum(x, axis=0), out.numpy()))


def _test_partial_sum_to_broadcast(test_case, src_device_type, dst_device_type):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    def build_p2b(input_blob, src_device_num, dst_device_num):
        with flow.scope.placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(input_blob.with_distribute(flow.distribute.split(0)))
            src = flow.math.reduce_sum(src, axis=0)
        with flow.scope.placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.broadcast()))
        return dst

    @flow.global_function(function_config=func_config)
    def partial_sum_to_broadcast_job(input_blob: oft.Numpy.Placeholder((96, 96, 96))):
        result_list = []
        for i in (2, 3):
            for j in (1, 2, 3):
                result_list.append(build_p2b(input_blob, i, j))
        return tuple(result_list)

    x = np.random.uniform(-1e-5, 1e-5, (96, 96, 96)).astype(np.float32)
    result_tuple = partial_sum_to_broadcast_job(x).get()
    for out in result_tuple:
        test_case.assertTrue(np.allclose(np.sum(x, axis=0), out.numpy()))


def _test_broadcast_to_broadcast(test_case, src_device_type, dst_device_type):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    def build_b2b(input_blob, src_device_num, dst_device_num):
        with flow.scope.placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(input_blob.with_distribute(flow.distribute.broadcast()))
        with flow.scope.placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.broadcast()))
        return dst

    @flow.global_function(function_config=func_config)
    def broadcast_to_broadcast_job(input_blob: oft.Numpy.Placeholder((96, 96))):
        result_list = []
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                result_list.append(build_b2b(input_blob, i, j))
        return tuple(result_list)

    x = np.random.rand(96, 96).astype(np.float32)
    result_tuple = broadcast_to_broadcast_job(x).get()
    for out in result_tuple:
        test_case.assertTrue(np.array_equal(x, out.numpy()))


def _test_multi_lbi(
    test_case, src_device_type, dst_device_type, src_device_num, dst_device_num
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    @flow.global_function(function_config=func_config)
    def multi_lbi_job(x: oft.Numpy.Placeholder((96, 96, 96))):
        with flow.scope.placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src_s0 = flow.identity(x.with_distribute(flow.distribute.split(0)))
            src_s1 = flow.identity(x.with_distribute(flow.distribute.split(1)))
            src_b = flow.identity(x.with_distribute(flow.distribute.split(1)))
            (t0_0, t0_1, t0_2) = flow.identity_n((src_s0, src_s1, src_b))
        with flow.scope.placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            t0_0 = t0_0.with_distribute(flow.distribute.split(1))
            t0_1 = t0_1.with_distribute(flow.distribute.broadcast())
            t0_2 = t0_2.with_distribute(flow.distribute.split(1))
            (t1_0, t1_1, t1_2) = flow.identity_n((t0_0, t0_1, t0_2))
        return t1_0, t1_1, t1_2

    x = np.random.uniform(-1e-5, 1e-5, (96, 96, 96)).astype(np.float32)
    r0 = multi_lbi_job(x).get()[0].numpy()
    r1 = multi_lbi_job(x).get()[1].numpy()
    r2 = multi_lbi_job(x).get()[2].numpy()
    test_case.assertTrue(np.array_equal(x, r0))
    test_case.assertTrue(np.array_equal(x, r1))
    test_case.assertTrue(np.array_equal(x, r2))


@flow.unittest.skip_unless_1n4d()
class TestBoxingV2(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_split_to_split(test_case):
        arg_dict = OrderedDict()
        arg_dict["src_device_type"] = ["cpu", "gpu"]
        arg_dict["dst_device_type"] = ["cpu", "gpu"]
        arg_dict["src_axis"] = [0, 1]
        arg_dict["dst_axis"] = [0, 1]
        for arg in GenArgList(arg_dict):
            _test_split_to_split(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_split_to_split_all_to_all(test_case):
        arg_dict = OrderedDict()
        arg_dict["src_device_type"] = ["gpu"]
        arg_dict["dst_device_type"] = ["gpu"]
        arg_dict["src_device_num"] = [4]
        arg_dict["dst_device_num"] = [4]
        for arg in GenArgList(arg_dict):
            _test_split_to_split_enable_all_to_all(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_split_to_broadcast(test_case):
        arg_dict = OrderedDict()
        arg_dict["src_device_type"] = ["cpu", "gpu"]
        arg_dict["dst_device_type"] = ["cpu", "gpu"]
        arg_dict["src_axis"] = [0, 1]
        for arg in GenArgList(arg_dict):
            _test_split_to_broadcast(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_broadcast_to_split(test_case):
        arg_dict = OrderedDict()
        arg_dict["src_device_type"] = ["cpu", "gpu"]
        arg_dict["dst_device_type"] = ["cpu", "gpu"]
        arg_dict["dst_axis"] = [0, 1]
        for arg in GenArgList(arg_dict):
            _test_broadcast_to_split(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_partial_sum_to_split(test_case):
        arg_dict = OrderedDict()
        arg_dict["src_device_type"] = ["cpu", "gpu"]
        arg_dict["dst_device_type"] = ["cpu", "gpu"]
        arg_dict["dst_axis"] = [0, 1]
        for arg in GenArgList(arg_dict):
            _test_partial_sum_to_split(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_partial_sum_to_broadcast(test_case):
        arg_dict = OrderedDict()
        arg_dict["src_device_type"] = ["cpu", "gpu"]
        arg_dict["dst_device_type"] = ["cpu", "gpu"]
        for arg in GenArgList(arg_dict):
            _test_partial_sum_to_broadcast(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_broadcast_to_broadcast(test_case):
        arg_dict = OrderedDict()
        arg_dict["src_device_type"] = ["cpu", "gpu"]
        arg_dict["dst_device_type"] = ["cpu", "gpu"]
        for arg in GenArgList(arg_dict):
            _test_broadcast_to_broadcast(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_multi_lbi(test_case):
        arg_dict = OrderedDict()
        arg_dict["src_device_type"] = ["cpu", "gpu"]
        arg_dict["dst_device_type"] = ["cpu", "gpu"]
        arg_dict["src_device_num"] = [1, 2, 3]
        arg_dict["dst_device_num"] = [1, 2, 3]
        for arg in GenArgList(arg_dict):
            _test_multi_lbi(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
