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

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest

# NOTE(Li Xiang): This variable controls the mem comparison method of the tensor offload test.
#  1: Strictly test, compare mem changes according to tensor size.
#  2: Loose test, compare mem changes before and after offload;
#  3: Execute only offload, skip mem check.
offload_tensor_test_mem_mode = 3


def _test_global_tensor_offload_d2h(test_case, input, tensor_mem):
    test_case.assertTrue(not input.is_offloaded())
    flow.cuda.empty_cache()
    if input.placement == oneflow.placement(type="cuda", ranks=[0, 1]):
        flow._oneflow_internal.CudaSynchronize(0)
        flow._oneflow_internal.CudaSynchronize(1)
    elif input.placement == oneflow.placement(type="cuda", ranks=[0, 1, 2, 3]):
        flow._oneflow_internal.CudaSynchronize(0)
        flow._oneflow_internal.CudaSynchronize(1)
        flow._oneflow_internal.CudaSynchronize(2)
        flow._oneflow_internal.CudaSynchronize(3)

    flow._oneflow_internal.eager.ClusterSync()
    before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    before_id = id(input)
    print("cuda", before_used)

    input.offload()
    test_case.assertTrue(input.is_offloaded())
    test_case.assertEqual(input.placement.type, "cuda")
    after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    after_id = id(input)
    print("cuda to cpu", after_used)
    # Check global_tensor_mem cuda memory released
    if offload_tensor_test_mem_mode == 1:
        # NOTE(Li Xiang): In the case of 4 gpus, the memory usage of the tensor sometimes has a 2MB error.
        if input.placement == oneflow.placement(type="cuda", ranks=[0, 1, 2, 3]):
            test_case.assertTrue(
                ((before_used - after_used) == tensor_mem)
                or ((before_used - after_used) == (tensor_mem - 2))
            )
            return
        test_case.assertTrue((before_used - after_used) == tensor_mem)
    elif offload_tensor_test_mem_mode == 2:
        test_case.assertTrue(before_used > after_used)
    elif offload_tensor_test_mem_mode == 3:
        print(
            "Device:",
            flow.env.get_rank(),
            ". cuda mem change value:",
            before_used - after_used,
        )
    test_case.assertEqual(before_id, after_id)


def _test_global_tensor_load_h2d(test_case, input, tensor_mem):
    test_case.assertTrue(input.is_offloaded())

    if input.placement == oneflow.placement(type="cuda", ranks=[0, 1]):
        flow._oneflow_internal.CudaSynchronize(0)
        flow._oneflow_internal.CudaSynchronize(1)
    elif input.placement == oneflow.placement(type="cuda", ranks=[0, 1, 2, 3]):
        flow._oneflow_internal.CudaSynchronize(0)
        flow._oneflow_internal.CudaSynchronize(1)
        flow._oneflow_internal.CudaSynchronize(2)
        flow._oneflow_internal.CudaSynchronize(3)

    flow._oneflow_internal.eager.ClusterSync()
    before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    before_id = id(input)

    input.load()
    test_case.assertTrue(not input.is_offloaded())
    test_case.assertEqual(input.placement.type, "cuda")
    after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    after_id = id(input)
    print("cpu to cuda", after_used)
    # Check global_tensor_mem cuda memory allocated
    if offload_tensor_test_mem_mode == 1:
        # NOTE(Li Xiang): In the case of 4 gpus, the memory usage of the tensor sometimes has a 2MB error.
        if input.placement == oneflow.placement(type="cuda", ranks=[0, 1, 2, 3]):
            test_case.assertTrue(
                ((after_used - before_used) == tensor_mem)
                or ((after_used - before_used) == (tensor_mem - 2))
            )
            return
        test_case.assertTrue((after_used - before_used) == tensor_mem)
    elif offload_tensor_test_mem_mode == 2:
        test_case.assertTrue(after_used > before_used)
    elif offload_tensor_test_mem_mode == 3:
        print(
            "Device:",
            flow.env.get_rank(),
            ". cuda mem change value:",
            after_used - before_used,
        )
    test_case.assertEqual(before_id, after_id)


def _get_specific_global_tensor_mem(placement, sbp, tensor):
    size_tensor = tensor.clone().detach().to_local()
    cnt_size = size_tensor.element_size() * flow.numel(size_tensor)
    return cnt_size / 1024 / 1024


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGlobalTensorOffload(flow.unittest.TestCase):
    @globaltest
    @flow.unittest.skip_unless_1n2d()
    def test_global_tensor_offload_and_load_2d(test_case):
        for i in range(5):
            placement = flow.placement("cuda", ranks=[0, 1])
            for sbp in all_sbp(placement, max_dim=2, except_partial_sum=True):
                input = flow.randn(
                    1024, 1024, 100, dtype=flow.float32, placement=placement, sbp=sbp
                )
                data = input.numpy()
                tensor_mem = _get_specific_global_tensor_mem(placement, sbp, input)
                _test_global_tensor_offload_d2h(test_case, input, tensor_mem)
                _test_global_tensor_load_h2d(test_case, input, tensor_mem)
                test_case.assertTrue(
                    np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001)
                )

    @globaltest
    @flow.unittest.skip_unless_1n4d()
    def test_global_tensor_offload_and_load_4d(test_case):
        for i in range(5):
            placement = flow.placement("cuda", ranks=[0, 1, 2, 3])
            for sbp in all_sbp(placement, max_dim=2, except_partial_sum=True):
                input = flow.randn(
                    1024, 1024, 10, dtype=flow.float32, placement=placement, sbp=sbp
                )
                data = input.numpy()
                tensor_mem = _get_specific_global_tensor_mem(placement, sbp, input)
                _test_global_tensor_offload_d2h(test_case, input, tensor_mem)
                _test_global_tensor_load_h2d(test_case, input, tensor_mem)
                test_case.assertTrue(
                    np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001)
                )

    @globaltest
    @flow.unittest.skip_unless_1n2d()
    def test_global_tensor_offload_and_load_2d_cpu_mem(test_case):
        flow.cuda.empty_cache()
        for i in range(5):
            placement = flow.placement("cuda", ranks=[0, 1])
            for sbp in all_sbp(placement, max_dim=2, except_partial_sum=True):
                input = flow.randn(
                    1024, 1024, 100, dtype=flow.float32, placement=placement, sbp=sbp
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

    @globaltest
    @flow.unittest.skip_unless_1n2d()
    def test_global_param_offload_and_load(test_case):
        def load_eager_model(model):
            for param in model.parameters():
                if param.is_offloaded():
                    param.load()
                    test_case.assertTrue(not param.is_offloaded())

        def offload_eager_model(model):
            for param in model.parameters():
                if not param.is_offloaded():
                    param.offload()
                    test_case.assertTrue(param.is_offloaded())

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_layer = 1

                layer_list = list()

                for _ in range(self.n_layer):
                    layer_list.append(nn.Linear(768, 4096))

                self.layers = nn.Sequential(*layer_list)

            def forward(self, x):
                return self.layers(x)

        placement = flow.placement("cuda", ranks=[0, 1])
        model0 = Model().cuda()
        model0.to_global(placement=placement, sbp=flow.sbp.broadcast)
        BZ = 128
        dataset = [flow.rand((BZ, 768), dtype=flow.float32) for _ in range(128)]

        with flow.no_grad():
            for idx, x in enumerate(dataset):
                print(f"iter {idx} begin")
                x = x.cuda()
                x = x.to_global(placement=placement, sbp=flow.sbp.broadcast)
                load_eager_model(model0)
                y0 = model0(x)
                offload_eager_model(model0)
                print(f"iter {idx} end")
                if idx == 1:
                    break


if __name__ == "__main__":
    unittest.main()
