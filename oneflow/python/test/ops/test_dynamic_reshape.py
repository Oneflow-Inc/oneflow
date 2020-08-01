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


def test_dynamic_reshape(test_case):
    num_gpus = 2
    data_shape = (10, 10, 10)
    flow.config.gpu_device_num(num_gpus)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(func_config)
    def DynamicReshapeJob(x: oft.ListNumpy.Placeholder(data_shape)):
        reshape_out1 = flow.reshape(x, (-1, 20))
        my_model = flow.get_variable(
            "my_model",
            shape=(20, 32),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
            trainable=True,
        )
        my_model = flow.cast_to_current_logical_view(my_model)
        mm_out = flow.matmul(reshape_out1, my_model)
        reshape_out2 = flow.reshape(mm_out, (-1, 8, 4))
        flow.losses.add_loss(reshape_out2)
        return reshape_out1

    data = [np.random.rand(*data_shape).astype(np.float32) for i in range(num_gpus)]
    out = DynamicReshapeJob(data).get().numpy_list()
    for i in range(num_gpus):
        test_case.assertTrue(np.array_equal(np.reshape(data[i], (50, 20)), out[i]))
