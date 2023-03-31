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
import os
import unittest

import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest

# NOTE(Li Xiang): This variable controls the mem comparison method of the tensor offload test.
#  1: Strictly test, compare mem changes according to tensor size.
#  2: Loose test, compare mem changes before and after offload;
#  3: Execute only offload, skip mem check.
offload_tensor_test_mem_mode = 3


def _test_tensor_offload_d2h(test_case, input, tensor_mem):
    print("\n- test offload cuda mem use")
    test_case.assertTrue(not input.is_offloaded())

    before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    print("  - before ", before_used)
    before_id = id(input)

    input.offload()
    test_case.assertTrue(input.is_offloaded())
    test_case.assertEqual(input.device, flow.device("cuda"))
    after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    after_id = id(input)
    print("  - after ", after_used)
    change_as_expected = (before_used - after_used) == tensor_mem
    # Check tensor_mem cuda memory released
    if offload_tensor_test_mem_mode == 1:
        test_case.assertTrue(change_as_expected)
    elif offload_tensor_test_mem_mode == 2:
        if tensor_mem != 0:
            test_case.assertTrue(before_used > after_used)
    print("  - tensor size ", tensor_mem)
    print("  - change ", after_used - before_used)
    print("  - change as expected ", change_as_expected)
    test_case.assertEqual(before_id, after_id)


def _test_tensor_load_h2d(test_case, input, tensor_mem):
    print("\n- test load cuda mem use")
    test_case.assertTrue(input.is_offloaded())

    before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    print("  - before ", before_used)
    before_id = id(input)

    input.load()
    test_case.assertTrue(not input.is_offloaded())
    test_case.assertEqual(input.device, flow.device("cuda"))
    after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    after_id = id(input)
    print("  - after ", after_used)
    # Check tensor_mem cuda memory allocated
    change_as_expected = (after_used - before_used) == tensor_mem
    if offload_tensor_test_mem_mode == 1:
        test_case.assertTrue(change_as_expected)
    elif offload_tensor_test_mem_mode == 2:
        if tensor_mem != 0:
            test_case.assertTrue(after_used > before_used)
    print("  - tensor size ", tensor_mem)
    print("  - change ", after_used - before_used)
    print("  - change as expected ", change_as_expected)
    test_case.assertEqual(before_id, after_id)


def _get_tensor_mem(input):
    if input.dim() == 0:
        return 2
    cnt_size = input.element_size() * flow.numel(input)
    return cnt_size / 1024 / 1024


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTensorOffload(flow.unittest.TestCase):
    def test_tensor_offload_and_load_float32(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randn(1024, 1024, 100),
            dtype=flow.float32,
            device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = _get_tensor_mem(input)
            # test tensor offload
            _test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            _test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    def test_tensor_offload_and_load_float16(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randn(20, 1024, 1024),
            dtype=flow.float16,
            device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = _get_tensor_mem(input)
            # test tensor offload
            _test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            _test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    def test_tensor_offload_and_load_int64(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randn(20, 1024, 1024),
            dtype=flow.int64,
            device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = _get_tensor_mem(input)
            # test tensor offload
            _test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            _test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    @unittest.skip("0 dim tensor is unstable in CI container mem tests.")
    def test_tensor_offload_and_load_0dim(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randint(1, 10), dtype=flow.float16, device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = _get_tensor_mem(input)
            # test tensor offload
            _test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            _test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    def test_tensor_offload_and_load_0size(test_case):
        flow.cuda.empty_cache()
        input = flow.tensor(
            np.random.randn(0, 1024, 1024),
            dtype=flow.float16,
            device=flow.device("cuda"),
        )
        data = input.numpy()

        for i in range(3):
            input_tensor_mem = 0
            # test tensor offload
            _test_tensor_offload_d2h(test_case, input, input_tensor_mem)

            # data = input.numpy() will raise error here

            # test tensor load
            _test_tensor_load_h2d(test_case, input, input_tensor_mem)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))

    def test_tensor_offload_and_load_cpu_mem(test_case):
        input = flow.tensor(
            np.random.randn(1024, 1024, 100),
            dtype=flow.float32,
            device=flow.device("cuda"),
        )

        before_used = flow._oneflow_internal.GetCPUMemoryUsed()
        before_id = id(input)
        input.offload()
        after_used = flow._oneflow_internal.GetCPUMemoryUsed()
        after_id = id(input)
        if offload_tensor_test_mem_mode == 2:
            test_case.assertTrue(after_used > before_used)
        elif offload_tensor_test_mem_mode == 3:
            print("cpu mem change value:", after_used - before_used)
        test_case.assertEqual(before_id, after_id)

        cur_used = flow._oneflow_internal.GetCPUMemoryUsed()
        before_id = id(input)
        input.load()
        after_used = flow._oneflow_internal.GetCPUMemoryUsed()
        after_id = id(input)
        if offload_tensor_test_mem_mode == 2:
            test_case.assertTrue(after_used < cur_used)
        elif offload_tensor_test_mem_mode == 3:
            print("cpu mem change value:", cur_used - after_used)
        test_case.assertEqual(before_id, after_id)

    def test_param_offload(test_case):
        def load_eager_model(model):
            for param in model.parameters():
                print("\n- test param load cuda mem use")
                test_case.assertTrue(param.is_offloaded())
                before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
                print("  - before ", before_used)
                param.load()
                after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
                print("  - after ", after_used)
                tensor_mem = _get_tensor_mem(param)
                change_as_expected = (after_used - before_used) == tensor_mem
                print("  - tensor size ", tensor_mem)
                print("  - change ", after_used - before_used)
                print("  - change as expected ", change_as_expected)
                test_case.assertTrue(not param.is_offloaded())

        def offload_eager_model(model):
            for param in model.parameters():
                print("\n- test param offload cuda mem use")
                test_case.assertTrue(not param.is_offloaded())
                before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
                print("  - before ", before_used)
                param.offload()
                after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
                print("  - after ", after_used)
                tensor_mem = _get_tensor_mem(param)
                change_as_expected = (before_used - after_used) == tensor_mem
                print("  - tensor size ", tensor_mem)
                print("  - change ", after_used - before_used)
                print("  - change as expected ", change_as_expected)
                test_case.assertTrue(param.is_offloaded())

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_layer = 1

                layer_list = list()

                for _ in range(self.n_layer):
                    # Too small to seem mem change
                    layer_list.append(nn.Linear(768, 4096))
                    # Big enough to seem mem change
                    layer_list.append(nn.Linear(4096, 4096))

                self.layers = nn.Sequential(*layer_list)

            def forward(self, x):
                return self.layers(x)

        model0 = Model().cuda()
        BZ = 128
        dataset = [flow.rand((BZ, 768), dtype=flow.float32) for _ in range(128)]

        with flow.no_grad():
            for idx, x in enumerate(dataset):
                print(f"iter {idx} begin")
                x = x.cuda()

                if idx != 0:
                    # no need to load at first iter
                    load_eager_model(model0)
                y0 = model0(x)
                offload_eager_model(model0)

                print(f"iter {idx} end")
                if idx == 1:
                    break


if __name__ == "__main__":
    unittest.main()
