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
class CpuTrilFlatKernel final : public user_op::OpKernel {
 public:
  CpuTrilFlatKernel() = default;
  ~CpuTrilFlatKernel() override = default;

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
    int64_t matrix_size = num_rows * num_cols;
    int64_t p = 0;
    for (int64_t k = 0; k < shape.elem_cnt(); ++k) {
      int64_t offset_in_matrix = k % matrix_size;
      int64_t row = offset_in_matrix / num_cols;
      int64_t col = offset_in_matrix - num_cols * row;
      if (row + diagonal >= col) { y_dptr[p++] = x_dptr[k]; }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class CpuTrilFlatBackwardKernel final : public user_op::OpKernel {
 public:
  CpuTrilFlatBackwardKernel() = default;
  ~CpuTrilFlatBackwardKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto shape = dx->shape();
    const auto diagonal = ctx->Attr<int64_t>("diagonal");
    const int64_t num_rows = shape.At(shape.NumAxes() - 2);
    const int64_t num_cols = shape.At(shape.NumAxes() - 1);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    T* dx_dptr = dx->mut_dptr<T>();
    const T* dy_dptr = dy->dptr<T>();
    int64_t matrix_size = num_rows * num_cols;
    int64_t p = 0;
    // const T fill = 0;
    for (int64_t k = 0; k < shape.elem_cnt(); ++k) {
      int64_t offset_in_matrix = k % matrix_size;
      int64_t row = offset_in_matrix / num_cols;
      int64_t col = offset_in_matrix - num_cols * row;
      dx_dptr[k] = row + diagonal < col ? 0 : dy_dptr[p++];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_TRIL_FLAT_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("tril_flat")                                                       \
      .SetCreateFn<CpuTrilFlatKernel<dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                     \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("tril_flat_grad")                                                  \
      .SetCreateFn<CpuTrilFlatBackwardKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                     \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));
REGISTER_CPU_TRIL_FLAT_KERNEL(float)
REGISTER_CPU_TRIL_FLAT_KERNEL(double)
REGISTER_CPU_TRIL_FLAT_KERNEL(uint8_t)
REGISTER_CPU_TRIL_FLAT_KERNEL(int8_t)
REGISTER_CPU_TRIL_FLAT_KERNEL(int32_t)
REGISTER_CPU_TRIL_FLAT_KERNEL(int64_t)

}  // namespace oneflow
