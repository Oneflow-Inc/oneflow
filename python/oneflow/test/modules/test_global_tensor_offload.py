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
import oneflow.unittest


def _test_global_tensor_offload_d2h(test_case, input, tensor_mem):
    test_case.assertTrue(not input.is_offloaded())
    flow.cuda.empty_cache()
    if input.placement == oneflow.placement(type="cuda", ranks=[0, 1]):
        flow._oneflow_internal.CudaSynchronize(0)
        flow._oneflow_internal.CudaSynchronize(1)
    elif input.placement == oneflow.placement(type="cuda", ranks=[0, 1,2,3]):
        flow._oneflow_internal.CudaSynchronize(0)
        flow._oneflow_internal.CudaSynchronize(1)
        flow._oneflow_internal.CudaSynchronize(2)
        flow._oneflow_internal.CudaSynchronize(3)

    flow._oneflow_internal.eager.ClusterSync()
    before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    print("cuda", before_used)

    input.offload()
    test_case.assertTrue(input.is_offloaded())
    # test_case.assertEqual(input.device, flow.device("cuda"))
    flow.cuda.empty_cache()
    flow._oneflow_internal.eager.ClusterSync()
    after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    print("cuda to cpu", after_used)
    # Check tensor_mem cuda memory released
    print(before_used - after_used)
    test_case.assertTrue((before_used - after_used) == tensor_mem)


def _test_global_tensor_load_h2d(test_case, input, tensor_mem):
    flow.cuda.empty_cache()
    test_case.assertTrue(input.is_offloaded())

    flow._oneflow_internal.eager.ClusterSync()
    before_used = flow._oneflow_internal.GetCUDAMemoryUsed()

    input.load()
    test_case.assertTrue(not input.is_offloaded())
    # test_case.assertEqual(input.device, flow.device("cuda"))
    flow.cuda.empty_cache()
    flow._oneflow_internal.eager.ClusterSync()
    after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    print("cpu to cuda", after_used)
    # Check tensor_mem cuda memory allocated
    print(after_used - before_used)
    test_case.assertTrue((after_used - before_used) == tensor_mem)


def _get_specific_global_tensor_mem(placement,sbp):
    print(sbp)
    if sbp[0]==oneflow.sbp.broadcast:
        if placement ==oneflow.placement(type="cuda", ranks=[0, 1]):
            return 400
        elif placement ==oneflow.placement(type="cuda", ranks=[0, 1,2,3]):  
            return 398
    if sbp[0]==oneflow.sbp.split(dim=0) or sbp[0]==oneflow.sbp.split(dim=1):
        if placement ==oneflow.placement(type="cuda", ranks=[0, 1]):
            return 200
        elif placement ==oneflow.placement(type="cuda", ranks=[0, 1,2,3]):  
            return 98
            



@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGlobalTensorOffload(flow.unittest.TestCase):
    @globaltest
    @flow.unittest.skip_unless_1n2d()
    def test_tensor_offload_and_load_2d(test_case):
        placement=flow.placement("cuda", ranks=[0, 1])
        for sbp in all_sbp(placement, max_dim=2,except_partial_sum=True):
            input = flow.randn(1024, 1024, 100, dtype=flow.float32, placement=placement, sbp=sbp)
            data = input.numpy()
            tensor_mem =_get_specific_global_tensor_mem(placement,sbp)
            _test_global_tensor_offload_d2h(test_case, input, tensor_mem)
            _test_global_tensor_load_h2d(test_case, input, tensor_mem)


    @globaltest
    @flow.unittest.skip_unless_1n4d()
    def test_tensor_offload_and_load_4d(test_case):
        placement=flow.placement("cuda", ranks=[0, 1,2,3])
        for sbp in all_sbp(placement, max_dim=2,except_partial_sum=True):
            flow.cuda.empty_cache()
            input = flow.randn(1024, 1024, 100, dtype=flow.float32, placement=placement, sbp=sbp)
            data = input.numpy()
            tensor_mem =_get_specific_global_tensor_mem(placement,sbp)
            print(tensor_mem)
            _test_global_tensor_offload_d2h(test_case, input, tensor_mem)
            _test_global_tensor_load_h2d(test_case, input, tensor_mem)



if __name__ == "__main__":
    unittest.main()
