from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList


def _test_split_to_split(
    src_device_type, dst_device_type, src_device_num, dst_device_num, src_axis, dst_axis
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.use_boxing_v2(True)

    @flow.global_function(func_config)
    def split_to_split_job(x=flow.FixedTensorDef((96, 96))):
        with flow.fixed_placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(x.with_distribute(flow.distribute.split(src_axis)))
        with flow.fixed_placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.split(dst_axis)))
        return dst

    x = np.random.rand(96, 96).astype(np.float32)
    y = split_to_split_job(x).get().ndarray()
    assert np.array_equal(x, y)


def test_split_to_split(test_case):
    arg_dict = OrderedDict()
    arg_dict["src_device_type"] = ["cpu", "gpu"]
    arg_dict["dst_device_type"] = ["cpu", "gpu"]
    arg_dict["src_device_num"] = [1, 2, 3]
    arg_dict["dst_device_num"] = [1, 2, 3]
    arg_dict["src_axis"] = [0, 1]
    arg_dict["dst_axis"] = [0, 1]
    for arg in GenArgList(arg_dict):
        _test_split_to_split(*arg)


def _test_split_to_broadcast(
    src_device_type, dst_device_type, src_device_num, dst_device_num, src_axis
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.use_boxing_v2(True)

    @flow.global_function(func_config)
    def split_to_broadcast_job(x=flow.FixedTensorDef((96, 96))):
        with flow.fixed_placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(x.with_distribute(flow.distribute.split(src_axis)))
        with flow.fixed_placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.broadcast()))
        return dst

    x = np.random.rand(96, 96).astype(np.float32)
    y = split_to_broadcast_job(x).get().ndarray()
    assert np.array_equal(x, y)


def test_split_to_broadcast(test_case):
    arg_dict = OrderedDict()
    arg_dict["src_device_type"] = ["cpu", "gpu"]
    arg_dict["dst_device_type"] = ["cpu", "gpu"]
    arg_dict["src_device_num"] = [1, 2, 3]
    arg_dict["dst_device_num"] = [1, 2, 3]
    arg_dict["src_axis"] = [0, 1]
    for arg in GenArgList(arg_dict):
        _test_split_to_broadcast(*arg)


def _test_broadcast_to_split(
    src_device_type, dst_device_type, src_device_num, dst_device_num, dst_axis
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.use_boxing_v2(True)

    @flow.global_function(func_config)
    def broadcast_to_split_job(x=flow.FixedTensorDef((96, 96))):
        with flow.fixed_placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(x.with_distribute(flow.distribute.broadcast()))
        with flow.fixed_placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.split(dst_axis)))
        return dst

    x = np.random.rand(96, 96).astype(np.float32)
    y = broadcast_to_split_job(x).get().ndarray()
    assert np.array_equal(x, y)


def test_broadcast_to_split(test_case):
    arg_dict = OrderedDict()
    arg_dict["src_device_type"] = ["cpu", "gpu"]
    arg_dict["dst_device_type"] = ["cpu", "gpu"]
    arg_dict["src_device_num"] = [1, 2, 3]
    arg_dict["dst_device_num"] = [1, 2, 3]
    arg_dict["dst_axis"] = [0, 1]
    for arg in GenArgList(arg_dict):
        _test_broadcast_to_split(*arg)


def _test_partial_sum_to_split(
    src_device_type, dst_device_type, src_device_num, dst_device_num, dst_axis
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.use_boxing_v2(True)

    @flow.global_function(func_config)
    def partial_sum_to_split_job(x=flow.FixedTensorDef((96, 96, 96))):
        with flow.fixed_placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(x.with_distribute(flow.distribute.split(0)))
            src = flow.math.reduce_sum(src, axis=0)
        with flow.fixed_placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.split(dst_axis)))
        return dst

    x = np.random.uniform(-1e-5, 1e-5, (96, 96, 96)).astype(np.float32)
    y = partial_sum_to_split_job(x).get().ndarray()
    assert np.allclose(np.sum(x, axis=0), y)


def test_partial_sum_to_split(test_case):
    arg_dict = OrderedDict()
    arg_dict["src_device_type"] = ["cpu", "gpu"]
    arg_dict["dst_device_type"] = ["cpu", "gpu"]
    arg_dict["src_device_num"] = [2, 3]
    arg_dict["dst_device_num"] = [1, 2, 3]
    arg_dict["dst_axis"] = [0, 1]
    for arg in GenArgList(arg_dict):
        _test_partial_sum_to_split(*arg)


def _test_partial_sum_to_broadcast(
    src_device_type, dst_device_type, src_device_num, dst_device_num
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.use_boxing_v2(True)

    @flow.global_function(func_config)
    def partial_sum_to_broadcast_job(x=flow.FixedTensorDef((96, 96, 96))):
        with flow.fixed_placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(x.with_distribute(flow.distribute.split(0)))
            src = flow.math.reduce_sum(src, axis=0)
        with flow.fixed_placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.broadcast()))
        return dst

    x = np.random.uniform(-1e-5, 1e-5, (96, 96, 96)).astype(np.float32)
    y = partial_sum_to_broadcast_job(x).get().ndarray()
    assert np.allclose(np.sum(x, axis=0), y)


def test_partial_sum_to_broadcast(test_case):
    arg_dict = OrderedDict()
    arg_dict["src_device_type"] = ["cpu", "gpu"]
    arg_dict["dst_device_type"] = ["cpu", "gpu"]
    arg_dict["src_device_num"] = [2, 3]
    arg_dict["dst_device_num"] = [1, 2, 3]
    for arg in GenArgList(arg_dict):
        _test_partial_sum_to_broadcast(*arg)


def _test_broadcast_to_broadcast(
    src_device_type, dst_device_type, src_device_num, dst_device_num
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.use_boxing_v2(True)

    @flow.global_function(func_config)
    def broadcast_to_broadcast_job(x=flow.FixedTensorDef((96, 96, 96))):
        with flow.fixed_placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(x.with_distribute(flow.distribute.broadcast()))
        with flow.fixed_placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.broadcast()))
        return dst

    x = np.random.uniform(-1e-5, 1e-5, (96, 96, 96)).astype(np.float32)
    y = broadcast_to_broadcast_job(x).get().ndarray()
    assert np.array_equal(x, y)


def test_broadcast_to_broadcast(test_case):
    arg_dict = OrderedDict()
    arg_dict["src_device_type"] = ["cpu", "gpu"]
    arg_dict["dst_device_type"] = ["cpu", "gpu"]
    arg_dict["src_device_num"] = [1, 2, 3]
    arg_dict["dst_device_num"] = [1, 2, 3]
    for arg in GenArgList(arg_dict):
        _test_broadcast_to_broadcast(*arg)


def _test_multi_lbi(src_device_type, dst_device_type, src_device_num, dst_device_num):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.use_boxing_v2(True)

    @flow.global_function(func_config)
    def multi_lbi_job(x=flow.FixedTensorDef((96, 96, 96))):
        with flow.fixed_placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src_s0 = flow.identity(x.with_distribute(flow.distribute.split(0)))
            src_s1 = flow.identity(x.with_distribute(flow.distribute.split(1)))
            src_b = flow.identity(x.with_distribute(flow.distribute.split(1)))
            (t0_0, t0_1, t0_2) = flow.identity_n((src_s0, src_s1, src_b))
        with flow.fixed_placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            t0_0 = t0_0.with_distribute(flow.distribute.split(1))
            t0_1 = t0_1.with_distribute(flow.distribute.broadcast())
            t0_2 = t0_2.with_distribute(flow.distribute.split(1))
            (t1_0, t1_1, t1_2) = flow.identity_n((t0_0, t0_1, t0_2))
        return t1_0, t1_1, t1_2

    x = np.random.uniform(-1e-5, 1e-5, (96, 96, 96)).astype(np.float32)
    r0 = multi_lbi_job(x).get()[0].ndarray()
    r1 = multi_lbi_job(x).get()[1].ndarray()
    r2 = multi_lbi_job(x).get()[2].ndarray()
    assert np.array_equal(x, r0)
    assert np.array_equal(x, r1)
    assert np.array_equal(x, r2)


def test_multi_lbi(test_case):
    arg_dict = OrderedDict()
    arg_dict["src_device_type"] = ["cpu", "gpu"]
    arg_dict["dst_device_type"] = ["cpu", "gpu"]
    arg_dict["src_device_num"] = [1, 2, 3]
    arg_dict["dst_device_num"] = [1, 2, 3]
    for arg in GenArgList(arg_dict):
        _test_multi_lbi(*arg)
