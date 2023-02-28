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
import oneflow
import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList

import time
import os


def _test_nccl_logical_send_recv_2d(test_case, src_nd_sbp, dst_nd_sbp):
    # can not process p in dst
    if flow.sbp.partial_sum() in dst_nd_sbp:
        return

    # skip src == dst
    if src_nd_sbp == dst_nd_sbp:
        return

    # in this case, use intra group boxing
    if src_nd_sbp[0] == dst_nd_sbp[0]:
        return

    # in this case, use inter group boxing
    if (
        src_nd_sbp[1] == dst_nd_sbp[1]
        and src_nd_sbp[0] != src_nd_sbp[1]
        and dst_nd_sbp[0] != dst_nd_sbp[1]
    ):
        return

    # input
    placement = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
    local_np = np.arange(4 * 4 * 4).reshape(4, 4, 4)
    x = flow.tensor(local_np, sbp=src_nd_sbp, placement=placement)

    # check eager boxing
    eager_out = x.to_global(sbp=dst_nd_sbp, placement=placement)
    test_case.assertTrue(np.array_equal(eager_out.numpy(), x.numpy()))

    # check graph boxing
    flow.boxing.nccl.enable_use_compute_stream(True)

    class TestNcclLogicalSendRecv2DGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x):
            y = x.to_global(sbp=dst_nd_sbp, placement=placement)
            return y

    graph = TestNcclLogicalSendRecv2DGraph()
    # graph.debug()
    y = graph(x)
    out_np = y.numpy()
    in_np = x.numpy()
    # if flow.env.get_rank() == 0:
    #    print("src sbp ", src_nd_sbp, ", dst sbp ", dst_nd_sbp)
    #    equal = np.array_equal(out_np, in_np)
    #    if not equal:
    #        print("in ", in_np)
    #        print("out ", out_np)
    test_case.assertTrue(np.array_equal(out_np, in_np))


def gen_2d_sbp():
    sbp_list = [
        flow.sbp.partial_sum(),
        flow.sbp.broadcast(),
        flow.sbp.split(0),
        flow.sbp.split(1),
        flow.sbp.split(2),
    ]
    nd_sbp_list = []
    for sbp0 in sbp_list:
        for sbp1 in sbp_list:
            nd_sbp_list.append([sbp0, sbp1])
    return nd_sbp_list


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestNcclLogicalSendRecv2D(flow.unittest.TestCase):
    def test_nccl_logical_send_recv_2d(test_case):
        os.environ["ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK"] = "1"
        arg_dict = OrderedDict()
        arg_dict["src_nd_sbp"] = gen_2d_sbp()
        arg_dict["dst_nd_sbp"] = gen_2d_sbp()
        for arg in GenArgList(arg_dict):
            _test_nccl_logical_send_recv_2d(test_case, *arg)


def _test_nccl_logical_send_recv_1d(test_case, src_nd_sbp, dst_nd_sbp):
    # can not process p in dst
    if flow.sbp.partial_sum() in dst_nd_sbp:
        return

    # skip src == dst
    if src_nd_sbp == dst_nd_sbp:
        return

    # input
    placement = flow.placement("cuda", ranks=[0, 1])
    local_np = np.arange(2 * 2 * 2).reshape(2, 2, 2)
    x = flow.tensor(local_np, sbp=src_nd_sbp, placement=placement)

    # check eager boxing
    eager_out = x.to_global(sbp=dst_nd_sbp, placement=placement)
    test_case.assertTrue(np.array_equal(eager_out.numpy(), x.numpy()))

    # check graph boxing
    flow.boxing.nccl.enable_use_compute_stream(True)

    class TestNcclLogicalSendRecv1DGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x):
            y = x.to_global(sbp=dst_nd_sbp, placement=placement)
            return y

    graph = TestNcclLogicalSendRecv1DGraph()
    # graph.debug(0)
    y = graph(x)
    out_np = y.numpy()
    in_np = x.numpy()
    # if flow.env.get_rank() == 0:
    #    print("src sbp ", src_nd_sbp, ", dst sbp ", dst_nd_sbp)
    #    print(graph)
    #    equal = np.array_equal(out_np, in_np)
    #    if not equal:
    #        print("in ", in_np)
    #        print("out ", out_np)
    #    print("====================")
    test_case.assertTrue(np.array_equal(out_np, in_np))


def gen_1d_sbp():
    sbp_list = [
        flow.sbp.partial_sum(),
        flow.sbp.broadcast(),
        flow.sbp.split(0),
        flow.sbp.split(1),
        flow.sbp.split(2),
    ]
    nd_sbp_list = []
    for sbp0 in sbp_list:
        nd_sbp_list.append(
            [sbp0,]
        )
    return nd_sbp_list


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestNcclLogicalSendRecv1D(flow.unittest.TestCase):
    def test_nccl_logical_send_recv_1d(test_case):
        os.environ["ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK"] = "1"
        arg_dict = OrderedDict()
        arg_dict["src_nd_sbp"] = gen_1d_sbp()
        arg_dict["dst_nd_sbp"] = gen_1d_sbp()
        for arg in GenArgList(arg_dict):
            _test_nccl_logical_send_recv_1d(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
