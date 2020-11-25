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
import os
import shutil
from collections import OrderedDict

import time
import numpy as np
import oneflow as flow
import tensorflow as tf
import test_global_storage
from test_util import GenArgList


def compare_with_tensorflow(device_type, device_num, input_shape, shape):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    flow.config.gpu_device_num(device_num)
    case = device_type + '_' + str(device_num)# + '_' + '-'.join([str(e) for e in input_shape]) \
           #+ '_' + '-'.join([str(e) for e in shape])
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def ReshapeJob():
        with flow.scope.placement(device_type, "0:0-{}".format(device_num-1)):
            x = flow.get_variable(
                "in",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=2, maxval=5),
                trainable=True,
                distribute=flow.distribute.split(2)
            )

            loss = flow.reshape(x, shape)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            return x, loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    x, loss = ReshapeJob().get()

    path = os.path.join('log/snapshot', case)
    if os.path.exists(path):
        shutil.rmtree(path)

    print('save checkpoint to ', path)
    check_point.save(path)
    #print(x.numpy(), loss.numpy())
    time.sleep(5)
    # TensorFlow
    #with tf.GradientTape(persistent=True) as tape:
    #    x = tf.Variable(test_global_storage.Get("x"))
    #    tf_out = tf.reshape(x, shape)
    #loss_diff = test_global_storage.Get("loss_diff")
    #tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    #assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    #assert np.allclose(
    #    test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    #)


class TestReshapeV2(flow.unittest.TestCase):
    def test_reshape(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["device_num"] = [2]
        arg_dict["input_shape"] = [(5, 8, 16)]
        arg_dict["shape"] = [[-1, 16]]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
