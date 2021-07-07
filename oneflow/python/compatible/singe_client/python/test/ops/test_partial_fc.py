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
import numpy as np
import oneflow as flow
import oneflow.typing as oft
from collections import OrderedDict

from test_util import GenArgList
import test_global_storage
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type


def compare_with_np(device_type, label_type, num_classes, num_sample, batch_size):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.gpu_device_num(0)
        flow.config.cpu_device_num(4)
    else:
        flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.indexed_slices_optimizer_conf(dict(include_op_names=dict(op_name=[])))

    @flow.global_function(type="train", function_config=func_config)
    def PartialFcJob(
        labels: oft.Numpy.Placeholder(
            (batch_size,), dtype=type_name_to_flow_type[label_type]
        )
    ):
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x-weight",
                shape=(num_classes, 128),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
        with flow.scope.placement(device_type, "0:0-3"):
            lebels_distribute = flow.distribute.broadcast()
            weight_distribute = flow.distribute.split(0)
            (
                maped_label,
                sampled_label,
                sampled_weight,
            ) = flow.distributed_partial_fc_sample(
                weight=x.with_distribute(weight_distribute),
                label=labels.with_distribute(lebels_distribute),
                num_sample=num_sample,
            )
        with flow.scope.placement(device_type, "0:0"):
            sampled_weight = flow.identity(sampled_weight)
            loss = flow.math.square(sampled_weight)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch_diff(
                sampled_weight, test_global_storage.Setter("sampled_weight_diff")
            )
        return x, maped_label, sampled_label, sampled_weight

    # fake labels
    labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(
        type_name_to_np_type[label_type]
    )

    # OneFlow
    weight, maped_label, sampled_label, sampled_weight = PartialFcJob(labels).get()

    gpu_num = 4
    device_class_num = num_classes // gpu_num
    device_num_sample = num_sample // gpu_num
    global_sample_labels_list = []
    np_mapped_label = []
    label_map = {}
    for i in range(gpu_num):
        lower = i * device_class_num
        upper = (i + 1) * device_class_num
        condition = (labels >= lower) & (labels < upper)
        local_label = labels[condition]
        local_label = np.unique(local_label).astype(np.int32)

        idx_start = int(i * device_num_sample)
        idx_end = int((i + 1) * device_num_sample)
        local_sample_labels = sampled_label[idx_start:idx_end]
        global_sample_labels = local_sample_labels
        global_sample_labels_list.append(global_sample_labels)

        assert (
            np.all((local_sample_labels >= lower) & (local_sample_labels < upper))
            == True
        )
        assert len(local_sample_labels) == len(np.unique(local_sample_labels))
        assert (
            np.array_equal(local_label, global_sample_labels[0 : len(local_label)])
            == True
        )
        for j in range(len(global_sample_labels)):
            label_map[global_sample_labels[j]] = j + idx_start

    for i in range(len(labels)):
        np_mapped_label.append(label_map[labels[i]])
    assert np.array_equal(np.array(np_mapped_label), maped_label.numpy()) == True

    global_sample_label = np.array(global_sample_labels_list).flatten().astype(np.int32)
    np_sample_weight = weight[global_sample_label]
    assert np.array_equal(sampled_weight.numpy(), np_sample_weight) == True

    sampled_weight_diff = test_global_storage.Get("sampled_weight_diff")

    np_weight_diff = np.zeros(weight.shape)
    for i in range(len(global_sample_label)):
        np_weight_diff[global_sample_label[i]] = sampled_weight_diff[i]

    x_diff = test_global_storage.Get("x_diff")

    assert np.array_equal(test_global_storage.Get("x_diff"), np_weight_diff) == True


flow.clear_default_session()


@flow.unittest.skip_unless_1n4d()
class TestPartialFc(flow.unittest.TestCase):
    def test_partial_fc1(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["label_type"] = ["int32"]
        arg_dict["num_classes"] = [85744]
        arg_dict["num_sample"] = [8600]
        arg_dict["batch_size"] = [512]
        for arg in GenArgList(arg_dict):
            compare_with_np(*arg)

    def test_partial_fc2(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["label_type"] = ["int32"]
        arg_dict["num_classes"] = [200]
        arg_dict["num_sample"] = [64]
        arg_dict["batch_size"] = [32]
        for arg in GenArgList(arg_dict):
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
