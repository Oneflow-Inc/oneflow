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

import numpy as np

import oneflow
import oneflow as flow
import oneflow.framework.graph_build_util as graph_build_util
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphWithSysConf(flow.unittest.TestCase):
    def test_graph_config(test_case):
        flow.boxing.enable_fusion(True)

        flow.boxing.nccl.set_fusion_threshold_mbytes(800)
        flow.boxing.nccl.set_fusion_max_ops_num(10)
        flow.boxing.nccl.allow_fuse_all_reduce(True)
        flow.boxing.nccl.allow_fuse_reduce_scatter(True)
        flow.boxing.nccl.allow_fuse_all_gather(True)
        flow.boxing.nccl.allow_fuse_reduce(True)
        flow.boxing.nccl.allow_fuse_broadcast(True)
        flow.boxing.nccl.allow_fuse_mixed_ops(True)
        flow.boxing.nccl.enable_use_buffer_to_fuse_all_reduce(True)
        flow.boxing.nccl.set_stream_num(3)
        flow.boxing.nccl.enable_all_to_all(True)
        flow.boxing.nccl.enable_use_compute_stream(True)
        flow.boxing.nccl.disable_group_boxing_by_dst_parallel(True)

        flow.backends.cudnn.set_reserved_mem_mbytes(1000)
        flow.backends.cudnn.enable_fused_normalization_add_relu(True)

        flow.utils.load_library("")

        class CustomGraphSysConf(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                # amp
                self.config.enable_amp(True)
                grad_scaler = flow.amp.GradScaler(
                    init_scale=3000,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=1000,
                )
                self.set_grad_scaler(grad_scaler)

                self.config.allow_fuse_model_update_ops(True)
                self.config.allow_fuse_add_to_output(True)
                self.config.set_gradient_accumulation_steps(100)
                self.config.allow_fuse_cast_scale(True)
                self.config.enable_zero(True)
                self.config.enable_cudnn_conv_heuristic_search_algo(False)

            def build(self, x):
                return x

        g = CustomGraphSysConf()

        print("optimization conf: \n", g._optimization_conf_proto)
        test_case.assertTrue(g._optimization_conf_proto.nccl_use_compute_stream)
        g._generate_config_proto()
        print("graph conf: \n", g._config_proto)

        # nccl
        flow.boxing.nccl.enable_use_compute_stream(False)
        flow.boxing.nccl.disable_group_boxing_by_dst_parallel(False)
        flow.boxing.nccl.allow_fuse_all_reduce(False)
        flow.boxing.nccl.allow_fuse_broadcast(False)
        flow.utils.cpu_device_num(222)
        flow.utils.machine_num(999)
        flow.utils.comm_net_worker_num(111)
        flow.utils.max_mdsave_worker_num(88)
        flow.utils.compute_thread_pool_size(33)
        flow.utils.reserved_host_mem_mbyte(22)
        flow.utils.reserved_device_mem_mbyte(555)
        flow.utils.enable_debug_mode(True)
        flow.backends.cudnn.enable_fused_normalization_add_relu(False)
        flow.boxing.enable_fusion(False)

        test_case.assertTrue(not g._optimization_conf_proto.nccl_use_compute_stream)

        test_case.assertTrue(
            not g._optimization_conf_proto.disable_group_boxing_by_dst_parallel
        )

        test_case.assertTrue(
            not g._optimization_conf_proto.collective_boxing_conf.nccl_fusion_all_reduce
        )

        test_case.assertTrue(
            not g._optimization_conf_proto.collective_boxing_conf.nccl_fusion_broadcast
        )

        # utils
        test_case.assertTrue(g._optimization_conf_proto.machine_num == 999)

        test_case.assertTrue(g._optimization_conf_proto.cpu_device_num == 222)

        test_case.assertTrue(g._optimization_conf_proto.comm_net_worker_num == 111)

        test_case.assertTrue(g._optimization_conf_proto.max_mdsave_worker_num == 88)

        test_case.assertTrue(g._optimization_conf_proto.compute_thread_pool_size == 33)

        test_case.assertTrue(g._optimization_conf_proto.reserved_host_mem_mbyte == 22)

        test_case.assertTrue(
            g._optimization_conf_proto.reserved_device_mem_mbyte == 555
        )

        test_case.assertTrue(g._optimization_conf_proto.enable_debug_mode)

        # cudnn
        test_case.assertTrue(
            not g._optimization_conf_proto.cudnn_conf.enable_cudnn_fused_normalization_add_relu
        )

        # boxing
        test_case.assertTrue(
            not g._optimization_conf_proto.collective_boxing_conf.enable_fusion
        )

        print("optimization conf after session init: \n", g._optimization_conf_proto)


if __name__ == "__main__":
    unittest.main()
