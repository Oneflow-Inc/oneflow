/*
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
*/
#include "oneflow/extension/python/py_kernel_caller.h"
#include "oneflow/extension/python/py_compute.h"

namespace oneflow {
void PyForwardKernel::Compute(user_op::KernelComputeContext* ctx) const {
  ::oneflow::pyext::PyCompute(ctx, "forward");
}

void PyBackwardKernel::Compute(user_op::KernelComputeContext* ctx) const {
  ::oneflow::pyext::PyCompute(ctx, "backward");
}

}  // namespace oneflow
