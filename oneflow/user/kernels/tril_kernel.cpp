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
#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
class CpuTrilKernel final : public user_op::OpKernel {
 public:
  CpuTrilKernel() = default;
  ~CpuTrilKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto shape = x->shape();
    const auto diagonal = ctx->Attr<int64_t>("diagonal");
    const int64_t num_rows = shape.At(shape.NumAxes() - 2);
    const int64_t num_cols = shape.At(shape.NumAxes() - 1);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* y_dptr = y->mut_dptr<T>();
    const T* x_dptr = x->dptr<T>();
    const T fill = ctx->Attr<bool>("is_floating_fill_value")
                       ? static_cast<T>(ctx->Attr<double>("floating_fill_value"))
                       : static_cast<T>(ctx->Attr<int64_t>("integer_fill_value"));
    int64_t matrix_size = num_rows * num_cols;
    for (int64_t k = 0; k < shape.elem_cnt(); ++k) {
      int64_t offset_in_matrix = k % matrix_size;
      int64_t i = offset_in_matrix / num_cols;
      int64_t j = offset_in_matrix - num_cols * i;
      y_dptr[k] = j > i + diagonal ? fill : x_dptr[k];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_TRIL_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("tril").SetCreateFn<CpuTrilKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "cpu")                                            \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_TRIL_KERNEL(float)
REGISTER_CPU_TRIL_KERNEL(double)
REGISTER_CPU_TRIL_KERNEL(int8_t)
REGISTER_CPU_TRIL_KERNEL(int32_t)
REGISTER_CPU_TRIL_KERNEL(int64_t)

}  // namespace oneflow
