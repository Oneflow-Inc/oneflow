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
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.typing as oft

from test_util import GenArgList


def _gen_test_data(out_shape, segment_ids_shape, axis):
    data_shape = out_shape[:axis] + segment_ids_shape + out_shape[axis + 1 :]
    data = np.random.rand(*data_shape).astype(np.float32)
    segment_ids = np.random.randint(
        low=0, high=out_shape[axis], size=segment_ids_shape
    ).astype(np.int32)

    out = np.zeros(shape=out_shape, dtype=np.float32)
    if axis != 0:
        ref_perm = [axis] + list(range(0, axis)) + list(range(axis + 1, out.ndim))
        out = np.transpose(out, ref_perm)
        data_perm = (
            list(range(axis, axis + segment_ids.ndim))
            + list(range(0, axis))
            + list(range(axis + segment_ids.ndim, data.ndim))
        )
        data_copy = np.transpose(data, data_perm)
    else:
        data_copy = data
    for idx, i in np.ndenumerate(segment_ids):
        out[i] += data_copy[idx]
    if axis != 0:
        ref_perm = list(range(1, axis + 1)) + [0] + list(range(axis + 1, out.ndim))
        out = np.transpose(out, ref_perm)

    return data, segment_ids, out


def _test_unsorted_segment_sum_model_parallel_fw(
    test_case, device_type, out_shape, segment_ids_shape, axis, split_axis
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    data_arr, segment_ids_arr, out_arr = _gen_test_data(
        out_shape, segment_ids_shape, axis
    )

    @flow.global_function(function_config=func_config)
    def unsorted_segment_sum_job(
        data: oft.Numpy.Placeholder(data_arr.shape, dtype=flow.float),
        segment_ids: oft.Numpy.Placeholder(segment_ids_arr.shape, dtype=flow.int32),
        like: oft.Numpy.Placeholder(out_arr.shape, dtype=flow.float),
    ):
        with flow.scope.placement(device_type, "0:0-3"):
            if split_axis < axis:
                data = data.with_distribute(flow.distribute.split(split_axis))
            elif split_axis == axis:
                data = data.with_distribute(flow.distribute.broadcast())
            else:
                data = data.with_distribute(
                    flow.distribute.split(split_axis + len(segment_ids.shape) - 1)
                )
            segment_ids = segment_ids.with_distribute(flow.distribute.broadcast())
            like = like.with_distribute(flow.distribute.split(split_axis))
            if split_axis == axis:
                out0 = like
            else:
                out0 = flow.unsorted_segment_sum(
                    data=data,
                    segment_ids=segment_ids,
                    num_segments=out_shape[axis],
                    axis=axis,
                )
            out1 = flow.unsorted_segment_sum_like(
                data=data, segment_ids=segment_ids, like=like, axis=axis
            )
            return out0, out1

    out0, out1 = unsorted_segment_sum_job(data_arr, segment_ids_arr, out_arr).get()
    test_case.assertTrue(np.allclose(out0.numpy(), out_arr))
    test_case.assertTrue(np.allclose(out1.numpy(), out_arr))


@flow.unittest.skip_unless_1n4d()
class TestUnsortedSegmentSumModelParallel(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_unsorted_segment_sum_model_parallel_fw(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["out_shape"] = [(96, 96, 96)]
        arg_dict["segment_ids_shape"] = [(32, 48)]
        arg_dict["axis"] = [0, 1, 2]
        arg_dict["split_axis"] = [0, 1, 2]
        for arg in GenArgList(arg_dict):
            _test_unsorted_segment_sum_model_parallel_fw(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
