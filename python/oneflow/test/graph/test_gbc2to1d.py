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


def _test_general_basic_communication_2d_to_1d(test_case, src_nd_sbp, dst_nd_sbp):
    # can not process p in dst
    if flow.sbp.partial_sum() == dst_nd_sbp:
        return

    # input
    placement_x = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
    placement_y = flow.placement("cuda", ranks=[0, 3, 4])
    local_np = np.arange(13 * 5).reshape(13, 5)
    x = flow.tensor(local_np, sbp=src_nd_sbp, placement=placement_x)

    # check eager boxing
    eager_out = x.to_global(sbp=dst_nd_sbp, placement=placement_y)
    test_case.assertTrue(np.array_equal(eager_out.numpy(), x.numpy()))

    # check graph boxing
    flow.boxing.nccl.enable_use_compute_stream(False)

    class TestGeneralBasicCommunicationGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x):
            y = x.to_global(sbp=dst_nd_sbp, placement=placement_y)
            return y

    graph = TestGeneralBasicCommunicationGraph()
    y = graph(x)
    out_np = y.numpy()
    in_np = x.numpy()
    test_case.assertTrue(np.array_equal(out_np, in_np))


def gen_nd_sbp_1d():
    sbp_list = [
        flow.sbp.partial_sum(),
        flow.sbp.broadcast(),
        flow.sbp.split(0),
        flow.sbp.split(1),
    ]
    return sbp_list


def gen_nd_sbp_2d():
    nd_sbp_list = []
    for sbp0 in gen_nd_sbp_1d():
        for sbp1 in gen_nd_sbp_1d():
            nd_sbp_list.append([sbp0, sbp1])
    return nd_sbp_list


@flow.unittest.skip_unless_2n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGeneralBasicCommunication(flow.unittest.TestCase):
    def test_general_basic_communication(test_case):
        os.environ["ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK"] = "0"
        os.environ["ONEFLOW_BOXING_ENABLE_GENERAL_BASIC_COMMUNICATION"] = "1"

        arg_dict = OrderedDict()
        arg_dict["src_nd_sbp"] = gen_nd_sbp_2d()
        arg_dict["dst_nd_sbp"] = gen_nd_sbp_1d()
        for arg in GenArgList(arg_dict):
            _test_general_basic_communication_2d_to_1d(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
