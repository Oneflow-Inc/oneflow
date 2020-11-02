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
import oneflow.typing as oft

from test_util import GenArgList

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_logical_view(flow.scope.consistent_view())


def _check(test_case, data, segment_ids, out_shape, axis, out):
    test_case.assertEqual(out.shape, out_shape)
    ref = np.zeros_like(out)
    if axis != 0:
        ref_perm = [axis] + list(range(0, axis)) + list(range(axis + 1, ref.ndim))
        ref = np.transpose(ref, ref_perm)
        data_perm = (
            list(range(axis, axis + segment_ids.ndim))
            + list(range(0, axis))
            + list(range(axis + segment_ids.ndim, data.ndim))
        )
        data = np.transpose(data, data_perm)
    for idx, i in np.ndenumerate(segment_ids):
        ref[i] += data[idx]
    if axis != 0:
        ref_perm = list(range(1, axis + 1)) + [0] + list(range(axis + 1, ref.ndim))
        ref = np.transpose(ref, ref_perm)
    test_case.assertTrue(np.allclose(ref, out, rtol=1e-3, atol=1e-3))


def _gen_segment_ids(out_shape, axis, segment_ids_shape):
    return np.random.randint(0, out_shape[axis], tuple(segment_ids_shape)).astype(
        np.int32
    )


def _gen_data(out_shape, axis, segment_ids_shape):
    data_shape = out_shape[0:axis] + segment_ids_shape + out_shape[axis + 1 :]
    return np.random.rand(*data_shape).astype(np.float32)


def _run_test(test_case, device, out_shape, axis, segment_ids_shape):
    flow.clear_default_session()

    segment_ids = _gen_segment_ids(out_shape, axis, segment_ids_shape)
    data = _gen_data(out_shape, axis, segment_ids_shape)

    @flow.global_function(function_config=func_config)
    def unsorted_segment_sum_job(
        data: oft.Numpy.Placeholder(data.shape, dtype=flow.float),
        segment_ids: oft.Numpy.Placeholder(segment_ids.shape, dtype=flow.int32),
    ):
        with flow.scope.placement(device, "0:0"):
            return flow.math.unsorted_segment_sum(
                data=data,
                segment_ids=segment_ids,
                num_segments=out_shape[axis],
                axis=axis,
            )

    @flow.global_function(function_config=func_config)
    def unsorted_segment_sum_like_job(
        data: oft.Numpy.Placeholder(data.shape, dtype=flow.float),
        segment_ids: oft.Numpy.Placeholder(segment_ids.shape, dtype=flow.int32),
        like: oft.Numpy.Placeholder(out_shape, dtype=flow.float32),
    ):
        with flow.scope.placement(device, "0:0"):
            return flow.math.unsorted_segment_sum_like(
                data=data, segment_ids=segment_ids, like=like, axis=axis
            )

    out = unsorted_segment_sum_job(data, segment_ids).get()
    _check(test_case, data, segment_ids, out_shape, axis, out.numpy())

    like = np.zeros(out_shape, dtype=np.float32)
    out = unsorted_segment_sum_like_job(data, segment_ids, like).get()
    _check(test_case, data, segment_ids, out_shape, axis, out.numpy())


@flow.unittest.skip_unless_1n1d()
class TestUnsortedSegmentSum(flow.unittest.TestCase):
    def test_unsorted_segment_sum(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["out_shape"] = [(4,), (4, 5), (4, 5, 6), (4, 5, 6, 7)]
        arg_dict["axis"] = [0, 1, 2, 3]
        arg_dict["segment_ids_shape"] = [(64,), (64, 96)]
        for arg in GenArgList(arg_dict):
            # axis >= len(out_shape)
            if arg[2] >= len(arg[1]):
                continue
            _run_test(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
