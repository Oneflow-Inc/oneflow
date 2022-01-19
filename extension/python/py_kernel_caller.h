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
#ifndef ONEFLOW_EXTENSION_PYTHON_PY_KERNEL_CALLER_H_
#define ONEFLOW_EXTENSION_PYTHON_PY_KERNEL_CALLER_H_
#include "oneflow/core/framework/framework.h"

namespace oneflow {
class PyForwardKernel final : public user_op::OpKernel {
 public:
  PyForwardKernel() = default;
  ~PyForwardKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class PyBackwardKernel final : public user_op::OpKernel {
 public:
  PyBackwardKernel() = default;
  ~PyBackwardKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_EXTENSION_PYTHON_PY_KERNEL_CALLER_H_
