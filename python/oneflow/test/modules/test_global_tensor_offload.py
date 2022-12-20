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
import oneflow as flow
import numpy as np


placement = flow.placement("cuda", [0, 1])
sbp = flow.sbp.split(1)

input = flow.randn(1024, 1024, 100, dtype=flow.float32, placement=placement, sbp=sbp)

data = input.numpy()

# print(input.is_offloaded(),flow.cuda.current_device())


flow.cuda.empty_cache()

before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
if flow.cuda.current_device() == 0:
    print("cuda", before_used, flow.cuda.current_device())

input.offload()
# print(input.is_offloaded(),flow.cuda.current_device())
flow.cuda.empty_cache()

after_used = flow._oneflow_internal.GetCUDAMemoryUsed()

if flow.cuda.current_device() == 0:
    print("cuda to cpu", after_used, flow.cuda.current_device())
# Check tensor_mem cuda memory released
if flow.cuda.current_device() == 0:
    print("offload", before_used - after_used, flow.cuda.current_device())


print("-------------")
# print(input.is_offloaded(),flow.cuda.current_device())
before_used = flow._oneflow_internal.GetCUDAMemoryUsed()

input.load()
# print(input.is_offloaded(),flow.cuda.current_device())
flow.cuda.empty_cache()
after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
if flow.cuda.current_device() == 0:
    print("cpu to cuda", after_used, flow.cuda.current_device())
# Check tensor_mem cuda memory allocated
if flow.cuda.current_device() == 0:
    print("load", after_used - before_used, flow.cuda.current_device())
