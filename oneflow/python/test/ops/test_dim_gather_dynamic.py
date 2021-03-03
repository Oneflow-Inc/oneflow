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
import oneflow.typing as oft
import os


def gen_gather_test_sample(input_shape, index_shape, dim, is_float=True):
    def _np_dim_scatter_add(src, dim, index, outshape):
        output = np.zeros(outshape)
        for srcidx in range(0, src.size):
            outcoord = np.unravel_index(srcidx, src.shape)
            outcoord = [*outcoord]
            outcoord[dim] = index[np.unravel_index(srcidx, index.shape)]
            output_offset = np.ravel_multi_index(outcoord, outshape)
            output[np.unravel_index(output_offset, outshape)] += src[
                np.unravel_index(srcidx, src.shape)
            ]
        return output

    if is_float:
        input = np.random.random(input_shape)
    else:
        input = np.random.randint(0, 100, input_shape)
    index = np.random.randint(0, input_shape[dim], index_shape)
    output = np.take_along_axis(input, index, dim)
    grad = _np_dim_scatter_add(np.ones_like(output), dim, index, input_shape)

    ret = {"input": input, "index": index, "dim": dim, "output": output, "grad": grad}
    return ret


def _make_dim_gather_fn(test_case, sample, datashape):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_placement_scope(flow.scope.placement("gpu", "0:0"))

    def _compare_diff(blob: oft.ListNumpy):
        test_case.assertTrue(np.allclose(sample["grad"], blob[0]))

    @flow.global_function(type="train", function_config=func_config)
    def DynamicDimGatherJob(
        params_def: oft.ListNumpy.Placeholder(datashape, dtype=flow.float32),
        index_def: oft.ListNumpy.Placeholder(datashape, dtype=flow.int32),
    ) -> oft.ListNumpy:
        x_var = flow.get_variable(
            "input",
            shape=(1,),
            dtype=flow.float32,
            initializer=flow.constant_initializer(0),
        )
        x_var = flow.cast_to_current_logical_view(x_var)
        x = x_var + params_def

        y = flow.dim_gather(x, sample["dim"], index_def)

        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(y)

        flow.watch_diff(x, _compare_diff)
        return y

    return DynamicDimGatherJob


def _compare_dim_gather_with_samples(test_case, inputshape, indexshape, dim, maxshape):
    sample = gen_gather_test_sample((inputshape), indexshape, dim)
    dynamic_dim_gather = _make_dim_gather_fn(test_case, sample, maxshape)
    out = dynamic_dim_gather([sample["input"]], [sample["index"]])[0]
    test_case.assertTrue(
        np.allclose(out, sample["output"].astype(np.float32), 1e-3, 1e-3)
    )


# @flow.unittest.skip_unless_1n1d()
# TODO(zhangwenxiao, jiangxuefei): refine in multi-client
@unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
class TestDynamicDimGather(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dynamic_dim_gather(test_case):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return
        _compare_dim_gather_with_samples(
            test_case, inputshape=(2, 2), indexshape=(2, 2), dim=1, maxshape=(10, 10)
        )

        _compare_dim_gather_with_samples(
            test_case, inputshape=(2, 2), indexshape=(2, 2), dim=0, maxshape=(10, 10)
        )

        _compare_dim_gather_with_samples(
            test_case,
            inputshape=(4, 4, 3),
            indexshape=(4, 1, 3),
            dim=1,
            maxshape=(10, 10, 10),
        )


if __name__ == "__main__":
    unittest.main()
