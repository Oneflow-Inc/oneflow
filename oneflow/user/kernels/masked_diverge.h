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
#ifndef _ONEFLOW_USER_KERNELS_MASKED_DIVERGE_H_
#define _ONEFLOW_USER_KERNELS_MASKED_DIVERGE_H_
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaskedDivergeKernel final : public user_op::OpKernel {
 public:
  MaskedDivergeKernel() = default;
  ~MaskedDivergeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_MASKED_DIVERGE_H_
