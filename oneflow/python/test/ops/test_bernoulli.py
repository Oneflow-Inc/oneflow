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
import oneflow.typing as oft


def test_bernoulli(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def BernoulliJob(a: oft.Numpy.Placeholder((10, 10))):
        return flow.random.bernoulli(a)

    x = np.ones((10, 10), dtype=np.float32)
    y = BernoulliJob(x).get().numpy()
    test_case.assertTrue(np.array_equal(y, x))

    x = np.zeros((10, 10), dtype=np.float32)
    y = BernoulliJob(x).get().numpy()
    test_case.assertTrue(np.array_equal(y, x))


def test_bernoulli_mirrored(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def BernoulliJob(a: oft.ListNumpy.Placeholder((10, 10))):
        return flow.random.bernoulli(a)

    x = np.ones((10, 10), dtype=np.float32)
    y = BernoulliJob([x]).get().numpy_list()[0]
    test_case.assertTrue(np.array_equal(y, x))

    x = np.zeros((10, 10), dtype=np.float32)
    y = BernoulliJob([x]).get().numpy_list()[0]
    test_case.assertTrue(np.array_equal(y, x))
