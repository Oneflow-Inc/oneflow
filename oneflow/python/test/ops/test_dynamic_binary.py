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
import numpy as np
import oneflow as flow
import oneflow.typing as oft


@flow.unittest.skip_unless_2n1d()
@unittest.skipIf(True, "always failed so skip it")
class TestDynamicBinary(flow.unittest.TestCase):
    def test_multi_node_dynamic_binary_split_concat_empty(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())
        func_config.default_placement_scope(flow.scope.placement("cpu", "0:0"))
        func_config.default_data_type(flow.float)
        flow.config.machine_num(2)
        flow.config.gpu_device_num(1)

        @flow.global_function(function_config=func_config)
        def DynamicBinaryJob(x: oft.ListNumpy.Placeholder((20,))):
            print("in_shape: ", x.shape)
            with flow.scope.placement("cpu", "0:0"):
                out_list = flow.experimental.dynamic_binary_split(
                    x, base_shift=4, out_num=6
                )
                id_out_list = []
                for out_blob in out_list:
                    print("out_shape: ", out_blob.shape)
                    id_out_list.append(flow.identity(out_blob))
            with flow.scope.placement("cpu", "1:0"):
                out1 = flow.experimental.dynamic_binary_concat(id_out_list, x)
                print("concat_shape: ", out1.shape)
            with flow.scope.placement("cpu", "0:0"):
                out2 = flow.identity(out1)
                print("return_shape: ", out2.shape)
            return out2

        size = [0, 5, 10, 15, 20]
        data = []
        for i in size:
            data.append(np.ones((i,), dtype=np.float32))
        for i in range(5):
            ret = DynamicBinaryJob([data[i]]).get().numpy_list()[0]
            print(ret)
            test_case.assertTrue(np.array_equal(ret, data[i]))


if __name__ == "__main__":
    unittest.main()
