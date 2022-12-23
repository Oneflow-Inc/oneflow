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

input = flow.tensor(
    np.random.randn(1024, 1024, 100), dtype=flow.float32, device=flow.device("cuda"),
)

b = flow._oneflow_internal.GetCPUMemoryUsed()
input.offload()
a = flow._oneflow_internal.GetCPUMemoryUsed()
print(a - b)


c = flow._oneflow_internal.GetCPUMemoryUsed()
input.load()
d = flow._oneflow_internal.GetCPUMemoryUsed()
print(c - d)
