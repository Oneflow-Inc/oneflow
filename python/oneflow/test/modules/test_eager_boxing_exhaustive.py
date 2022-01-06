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
import os

import oneflow.unittest
from test_util import GenArgList


def _test_eager_boxing_normal_1d_exhaustive_testing(
    test_case, shape, in_device, out_device, in_device_list, out_device_list
):
    import itertools

    sbps = [
        flow.sbp.split(0),
        flow.sbp.split(1),
        flow.sbp.broadcast,
        flow.sbp.partial_sum,
    ]
    np.random.seed(10)
    np_arr = np.random.uniform(-1e2, 1e2, shape)
    in_placement = flow.placement(in_device, {0: in_device_list})
    out_placement = flow.placement(out_device, {0: out_device_list})
    failed_boxing = []
    for elem in itertools.product(sbps, sbps):
        try:
            x = flow.tensor(
                np_arr,
                dtype=flow.float32,
                placement=in_placement,
                sbp=[elem[0]],
                requires_grad=False,
            )
            y = x.to_consistent(placement=out_placement, sbp=[elem[1]])

            z = y.to_consistent(placement=out_placement, sbp=[flow.sbp.broadcast])
            if flow.env.get_rank() in out_device_list:
                test_case.assertTrue(np.allclose(z.to_local().numpy(), np_arr),)
        except flow._oneflow_internal.exception.BoxingNotSupportedException:
            failed_boxing.append(
                (elem, shape, in_device, out_device, in_device_list, out_device_list)
            )

    if flow.env.get_rank() == 0:
        print(
            "%d unsuported boxing 1d type" % len(failed_boxing),
            failed_boxing,
            sep="\n",
        )


def _test_eager_boxing_symmetric_2d_exhaustive_testing(
    test_case, in_device, out_device
):
    import itertools

    sbps = [
        flow.sbp.split(0),
        flow.sbp.split(1),
        flow.sbp.broadcast,
        flow.sbp.partial_sum,
    ]
    nd_sbps = itertools.product(
        itertools.product(sbps, sbps), itertools.product(sbps, sbps)
    )
    np.random.seed(20)
    np_arr = np.random.uniform(-1e2, 1e2, (32, 96, 64))
    in_placement = flow.placement(in_device, {0: range(4)}, (2, 2))
    out_placement = flow.placement(out_device, {0: range(4)}, (2, 2))
    failed_boxing = []
    for elem in nd_sbps:
        try:
            x = flow.tensor(
                np_arr,
                dtype=flow.float32,
                placement=in_placement,
                sbp=elem[0],
                requires_grad=False,
            )
            y = x.to_consistent(placement=out_placement, sbp=elem[1])

            z = y.to_consistent(
                placement=out_placement, sbp=[flow.sbp.broadcast, flow.sbp.broadcast]
            )
            test_case.assertTrue(np.allclose(z.to_local().numpy(), np_arr,),)
        except flow._oneflow_internal.exception.UnimplementedException:
            failed_boxing.append(elem)

    if flow.env.get_rank() == 0:
        print(
            "%d unsuported boxing 2d type" % len(failed_boxing), failed_boxing, sep="\n"
        )


def _test_eager_boxing_1d_special_split_axis(
    test_case, in_device, out_device, in_device_list, out_device_list
):
    import itertools

    sbps = [
        flow.sbp.split(2),
        flow.sbp.split(4),
        flow.sbp.broadcast,
        flow.sbp.partial_sum,
    ]
    np.random.seed(30)
    shape = (16, 16, 5, 8, 7, 6)
    np_arr = np.random.uniform(-1e2, 1e2, shape)
    in_placement = flow.placement(in_device, {0: in_device_list})
    out_placement = flow.placement(out_device, {0: out_device_list})
    failed_boxing = []
    for elem in itertools.product(sbps, sbps):
        try:
            x = flow.tensor(
                np_arr,
                dtype=flow.float32,
                placement=in_placement,
                sbp=[elem[0]],
                requires_grad=False,
            )
            y = x.to_consistent(placement=out_placement, sbp=[elem[1]])

            z = y.to_consistent(placement=out_placement, sbp=[flow.sbp.broadcast])
            if flow.env.get_rank() in out_device_list:
                test_case.assertTrue(np.allclose(z.to_local().numpy(), np_arr),)
        except flow._oneflow_internal.exception.BoxingNotSupportedException:
            failed_boxing.append(
                (elem, shape, in_device, out_device, in_device_list, out_device_list)
            )

    if flow.env.get_rank() == 0:
        print(
            "%d unsuported boxing 1d type" % len(failed_boxing),
            failed_boxing,
            sep="\n",
        )


def _test_eager_boxing_2d_special_split_axis(
    test_case, in_device, out_device
):
    import itertools

    sbps = [
        flow.sbp.split(2),
        flow.sbp.split(4),
        flow.sbp.broadcast,
        flow.sbp.partial_sum,
    ]
    nd_sbps = itertools.product(
        itertools.product(sbps, sbps), itertools.product(sbps, sbps)
    )
    np.random.seed(40)
    np_arr = np.random.uniform(-1e2, 1e2, (8, 16, 4, 8, 12))
    in_placement = flow.placement(in_device, {0: range(4)}, (2, 2))
    out_placement = flow.placement(out_device, {0: range(4)}, (2, 2))
    failed_boxing = []
    for elem in nd_sbps:
        try:
            x = flow.tensor(
                np_arr,
                dtype=flow.float32,
                placement=in_placement,
                sbp=elem[0],
                requires_grad=False,
            )
            y = x.to_consistent(placement=out_placement, sbp=elem[1])

            z = y.to_consistent(placement=out_placement, sbp=[flow.sbp.broadcast, flow.sbp.broadcast])
            test_case.assertTrue(np.allclose(z.to_local().numpy(), np_arr),)
        except flow._oneflow_internal.exception.BoxingNotSupportedException:
            failed_boxing.append(
                (elem, in_device, out_device)
            )
        except flow._oneflow_internal.exception.UnimplementedException:
            failed_boxing.append(
                (elem, in_device, out_device)
            )

    if flow.env.get_rank() == 0:
        print(
            "%d unsuported boxing 2d type" % len(failed_boxing),
            failed_boxing,
            sep="\n",
        )


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingSymmetricExhaustiveTesting(flow.unittest.TestCase):
    def test_eager_boxing_normal_1d_exhaustive_testing(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(12, 12), (18, 24), (15, 17)]
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        arg_dict["in_device_list"] = [[0, 1], [1, 2, 3], [0, 1, 2, 3]]
        arg_dict["out_device_list"] = [[0, 1, 3], [0, 1, 2, 3]]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_normal_1d_exhaustive_testing(test_case, *arg)

    def test_eager_boxing_symmetric_2d_exhaustive_testing(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_symmetric_2d_exhaustive_testing(test_case, *arg)



@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingSpecialSplitAxisExhaustiveTesting(flow.unittest.TestCase):
    def test_eager_boxing_1d_special_split_axis(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        arg_dict["in_device_list"] = [[0, 1], [1, 2, 3], [0, 1, 2, 3]]
        arg_dict["out_device_list"] = [[0, 1, 3], [0, 1, 2, 3]]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_1d_special_split_axis(test_case, *arg)

    def test_eager_boxing_2d_special_split_axis(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_2d_special_split_axis(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
