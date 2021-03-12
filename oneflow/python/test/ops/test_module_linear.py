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
import numpy as np

import oneflow as flow
import oneflow.typing as tp


def compare_with_numpy(device_type, x_shape, in_features, out_features, bias):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    if data_type == "float16":
        dtype = flow.float
    else:
        dtype = type_name_to_flow_type[data_type]

    @flow.global_function(type="train", function_config=func_config)
    def LinearJob():
        with flow.scope.placement(device_type, "0:0"):
            linear = flow.nn.Linear(in_features, out_features, bias)
            x = flow.Tensor(2, 3)
            return linear(x)

    # OneFlow
    of_out = LinearJob().get()


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["x_shape"] = [(128, 20)]
    arg_dict["in_features"] = [20]
    arg_dict["out_features"] = [30]
    arg_dict["bias"] = [True, False]
    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestMatmul(flow.unittest.TestCase):
    def test_linear(test_case):
        for arg in gen_arg_list():
            compare_with_numpy(*arg)


if __name__ == "__main__":
    unittest.main()
