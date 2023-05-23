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

import oneflow._oneflow_internal


class cuMatmulMode:
    def __getattr__(self, name):
        if name == "allow_tf32":
            return oneflow._oneflow_internal.ep.is_matmul_allow_tf32()
        elif name == "allow_fp16_reduced_precision_reduction":
            return (
                oneflow._oneflow_internal.ep.is_matmul_allow_fp16_reduced_precision_reduction()
            )
        raise AssertionError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            return oneflow._oneflow_internal.ep.set_matmul_allow_tf32(value)
        elif name == "allow_fp16_reduced_precision_reduction":
            return oneflow._oneflow_internal.ep.set_matmul_allow_fp16_reduced_precision_reduction(
                value
            )
        raise AssertionError("Unknown attribute " + name)


matmul = cuMatmulMode()
