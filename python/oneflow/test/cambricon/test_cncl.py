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
import oneflow.unittest


@flow.unittest.skip_unless_1n2d()
class TestCnclCambriconModule(flow.unittest.TestCase):
    def test_cncl_all_reduce(test_case):
        arr = np.array(
            [
                [4.0, 6.0, 5.0, 20.0],
                [6.0, 2.0, 5.0, 7.0],
                [3.0, 7.0, 5.0, 4.0],
                [6.0, 8.0, 9.0, 4.0],
            ],
            dtype=np.float32,
        )
        arr_out = arr * 2

        x = flow.Tensor(arr, device="mlu")
        y = x.to_global(
            placement=flow.env.all_device_placement("mlu"), sbp=flow.sbp.partial_sum
        )
        eager_out = y.to_global(
            placement=flow.env.all_device_placement("mlu"), sbp=flow.sbp.broadcast
        )

        class AllReudce(flow.nn.Graph):
            def __init__(self):
                super(AllReudce, self).__init__()

            def build(self, x):
                y = x.to_global(
                    placement=flow.env.all_device_placement("mlu"),
                    sbp=flow.sbp.broadcast,
                )
                return y

        graph = AllReudce()
        lazy_out = graph(y)

        test_case.assertTrue(np.array_equal(eager_out.to_local().numpy(), arr_out))
        test_case.assertTrue(np.array_equal(lazy_out.to_local().numpy(), arr_out))

    def test_cncl_reduce_scatter(test_case):
        arr = np.array(
            [
                [4.0, 6.0, 5.0, 20.0],
                [6.0, 2.0, 5.0, 7.0],
                [3.0, 7.0, 5.0, 4.0],
                [6.0, 8.0, 9.0, 4.0],
            ],
            dtype=np.float32,
        )
        arr_out = arr * 2

        x = flow.Tensor(arr, device="mlu")
        y = x.to_global(
            placement=flow.env.all_device_placement("mlu"), sbp=flow.sbp.partial_sum
        )
        eager_out = y.to_global(
            placement=flow.env.all_device_placement("mlu"), sbp=flow.sbp.split(0)
        )

        class ReduceScatter(flow.nn.Graph):
            def __init__(self):
                super(ReduceScatter, self).__init__()

            def build(self, x):
                y = x.to_global(
                    placement=flow.env.all_device_placement("mlu"),
                    sbp=flow.sbp.split(0),
                )
                return y

        graph = ReduceScatter()
        lazy_out = graph(y)

        idx = flow.env.get_rank()
        step = 2

        test_case.assertTrue(
            np.array_equal(
                eager_out.to_local().numpy(), arr_out[idx * step : (idx + 1) * step]
            )
        )
        test_case.assertTrue(
            np.array_equal(
                lazy_out.to_local().numpy(), arr_out[idx * step : (idx + 1) * step]
            )
        )

    def test_cncl_all_gather(test_case):
        arr = np.array(
            [
                [4.0, 6.0, 5.0, 20.0],
                [6.0, 2.0, 5.0, 7.0],
                [3.0, 7.0, 5.0, 4.0],
                [6.0, 8.0, 9.0, 4.0],
            ],
            dtype=np.float32,
        )
        arr_out = np.concatenate((arr, arr), axis=0)

        x = flow.Tensor(arr, device="mlu")
        y = x.to_global(
            placement=flow.env.all_device_placement("mlu"), sbp=flow.sbp.split(0)
        )
        eager_out = y.to_global(
            placement=flow.env.all_device_placement("mlu"), sbp=flow.sbp.broadcast
        )

        class AllGather(flow.nn.Graph):
            def __init__(self):
                super(AllGather, self).__init__()

            def build(self, x):
                y = x.to_global(
                    placement=flow.env.all_device_placement("mlu"),
                    sbp=flow.sbp.broadcast,
                )
                return y

        graph = AllGather()
        lazy_out = graph(y)

        test_case.assertTrue(np.array_equal(eager_out.to_local().numpy(), arr_out))
        test_case.assertTrue(np.array_equal(lazy_out.to_local().numpy(), arr_out))

    def test_cncl_broadcast(test_case):
        arr = np.array(
            [
                [4.0, 6.0, 5.0, 20.0],
                [6.0, 2.0, 5.0, 7.0],
                [3.0, 7.0, 5.0, 4.0],
                [6.0, 8.0, 9.0, 4.0],
            ],
            dtype=np.float32,
        )
        arr_out = arr

        x = flow.Tensor(arr, device="mlu")
        y = x.to_global(
            placement=flow.env.all_device_placement("mlu"), sbp=flow.sbp.broadcast
        )
        eager_out = y.to_global(
            placement=flow.env.all_device_placement("mlu"), sbp=flow.sbp.broadcast
        )

        test_case.assertTrue(np.array_equal(eager_out.to_local().numpy(), arr_out))


if __name__ == "__main__":
    unittest.main()
