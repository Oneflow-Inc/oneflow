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
import oneflow.framework.session_context as session_ctx
import oneflow as flow
import oneflow.unittest
import oneflow.framework.config_util as config_util
import oneflow.framework.attr_util as attr_util
import random


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
        flow.backends.cudnn.enable_conv_heuristic_search_algo(False)

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

        # Test the resource config update eagerly
        # Note: this tests all the apis in oneflow.framework.config_util automatically
        def test_resource_config_update_apis_eagerly_automatically():
            attrs_and_values_to_check = []
            num_api_tested = 0

            for api in config_util.api_attrs_and_type.keys():
                attrs, type_ = config_util.api_attrs_and_type[api]
                if type_ is int:
                    attr_value = random.randint(0, 9999)
                    attrs_and_values_to_check.append((attrs, attr_value))
                elif type_ is bool:
                    attr_value = random.choice([True, False])
                    attrs_and_values_to_check.append((attrs, attr_value))
                else:
                    raise TypeError("Unsupported type!")

                api(attr_value)
                num_api_tested += 1

            # check all the attributes are set correctly
            for (attrs, expected_attr_value) in attrs_and_values_to_check:
                current_attr_value = attr_util.get_nested_attribute(
                    g._optimization_conf_proto, attrs
                )
                test_case.assertTrue(
                    current_attr_value == expected_attr_value,
                    str(attrs)
                    + " : "
                    + str(current_attr_value)
                    + " vs "
                    + str(current_attr_value),
                )

            print("number of APIs tested: " + str(num_api_tested))

        # save the resource config before running random resource api tests
        session = session_ctx.GetDefaultSession()
        prev_resource_config = session.resource

        for i in range(5):
            test_resource_config_update_apis_eagerly_automatically()

        print("optimization conf after session init: \n", g._optimization_conf_proto)

        # restore the resource config
        session.update_resource_eagerly(prev_resource_config)


if __name__ == "__main__":
    unittest.main()
