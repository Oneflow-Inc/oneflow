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


def test_deadlock(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.enable_inplace(False)

    @flow.global_function(function_config=func_config)
    def DistributeConcat():
        with flow.scope.placement("gpu", "0:0"):
            w = flow.get_variable(
                "w", (2, 5), initializer=flow.constant_initializer(10)
            )
            x = w + 1
            y = w + 1
        ret = flow.advanced.distribute_concat([x, y])
        # return ret

    DistributeConcat()
