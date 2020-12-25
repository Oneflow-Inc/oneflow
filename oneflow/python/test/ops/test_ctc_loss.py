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
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as tp

# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(
    device_type, max_logit_length, batch_size, num_classes, max_label_length, data_type
):
    assert device_type in ["gpu", "cpu"]
    assert data_type in ["float32", "double", "int8", "int32", "int64"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def ctc_loss_job(
        log_probs: tp.Numpy.Placeholder(
            shape=(max_logit_length, batch_size, num_classes)
        ),
        targets: tp.Numpy.Placeholder(
            shape=(batch_size, max_label_length), dtype=flow.int32
        ),
        input_lengths: tp.Numpy.Placeholder(shape=(batch_size,), dtype=flow.int32),
        target_lengths: tp.Numpy.Placeholder(shape=(batch_size,), dtype=flow.int32),
    ) -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            return flow.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, reduction="none"
            )

    log_probs = np.random.random(
        size=(max_logit_length, batch_size, num_classes)
    ).astype(type_name_to_np_type[data_type])
    log_probs = tf.nn.log_softmax(log_probs, axis=2)

    targets = np.random.randint(
        1, high=num_classes, size=(batch_size, max_label_length)
    )
    input_lengths = np.random.randint(
        max_logit_length / 2, high=max_logit_length, size=(batch_size,)
    )
    target_lengths = np.random.randint(
        max_label_length / 2, high=max_label_length, size=(batch_size,)
    )
    # print(log_probs)
    # print(targets)
    # print(input_lengths)
    # print(target_lengths)
    # OneFlow
    of_out = ctc_loss_job(log_probs.numpy(), targets, input_lengths, target_lengths)
    # TensorFlow
    tf_out = tf.nn.ctc_loss(targets, log_probs, target_lengths, input_lengths)

    print(of_out)
    print(tf_out.numpy())

    tolerance = 1e-5
    assert np.allclose(
        of_out, tf_out.numpy(), rtol=tolerance, atol=tolerance, equal_nan=True
    )


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu"]
    arg_dict["max_logit_length"] = [1000]  # Input sequence length
    arg_dict["batch_size"] = [10]  # Batch size
    arg_dict["num_classes"] = [10]  # Number of classes (including blank)
    arg_dict["max_label_length"] = [100]  # Target length of longest target in batch
    arg_dict["data_type"] = ["float32"]

    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestSort(flow.unittest.TestCase):
    def test_sort(test_case):
        for arg in gen_arg_list():
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
