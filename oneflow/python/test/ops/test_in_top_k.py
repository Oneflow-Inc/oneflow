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
import oneflow as flow
import numpy as np
import tensorflow as tf
import random


def compare(prediction, target, k):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    func_config.default_data_type(flow.float)

    def test_in_top_k(predictions, targets, k, name=None):
        if name is None:
            name = id_util.UniqueStr("intopk_")
        return (
            flow.user_op_builder(name)
            .Op("in_top_k")
            .Input("predictions", [predictions])
            .Input("targets", [targets])
            .SetAttr("k", k, "AttrTypeInt32",)
            .Output("out")
            .Build()
            .RemoteBlobList()[0]
        )

    @flow.function(func_config)
    def IntopkJob(
        prediction=flow.FixedTensorDef((3, 4), dtype=flow.float),
        target=flow.FixedTensorDef((3,), dtype=flow.int32),
    ):
        return flow.math.in_top_k(prediction, target, k=k)

    # OneFlow
    y = IntopkJob(prediction, target).get().ndarray()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        prediction_v = tf.Variable(prediction)
        target_v = tf.Variable(target)
        tf_out = tf.math.in_top_k(target_v, prediction_v, k=k)

    assert np.allclose(y, tf_out)


def test(test_case):

    test_time = 5
    for i in range(test_time):
        prediction = np.random.rand(3, 4).astype("float32")
        target = np.random.randint(4, size=3).astype("int32")
        k = random.choice([1, 2, 3, 4])
        compare(prediction, target, k)
