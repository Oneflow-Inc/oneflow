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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class CpuSpmmCsrKernel final : public user_op::OpKernel {
 public:
  CpuSpmmCsrKernel() = default;
  ~CpuSpmmCsrKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    CHECK_EQ(1, 2) << "spmm for cpu is not implemented yet ";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_SPMM_Csr_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("spmm_csr")                                    \
      .SetCreateFn<CpuSpmmCsrKernel<dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_SPMM_Csr_KERNEL(float)
REGISTER_CPU_SPMM_Csr_KERNEL(double)

}  // namespace oneflow