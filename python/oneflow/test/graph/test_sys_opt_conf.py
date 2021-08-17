import os
import unittest

import numpy as np

import oneflow
import oneflow as flow
import oneflow.framework.graph_build_util as graph_build_util
import oneflow.unittest


class SubModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = flow.nn.Conv2d(1, 1, 5)
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class CustomModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = SubModule()
        self.fc1 = flow.nn.Linear(36, 4)
        self.register_buffer("dummy_buff", flow.Tensor(1, 4))

    def forward(self, x):
        x = self.layer(x)
        x = oneflow.F.flatten(x, 1)
        x = self.fc1(x) + self.dummy_buff
        return x


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphWithSysConf(flow.unittest.TestCase):
    def test_graph_config(test_case):
        flow.backends.nccl.config.fusion_all_reduce_use_buffer(True)

        class CustomGraphSysConf(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()
                self.config.enable_auto_mixed_precision(True)
                self.config.enable_fuse_add_to_output(True)

            def build(self, x):
                x = self.m(x)
                return x

        g = CustomGraphSysConf()

        print("backends conf: \n", g._backends_conf_proto)
        print("graph conf: \n", g._graph_conf_proto)