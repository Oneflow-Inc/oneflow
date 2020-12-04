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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void TrilGpu(const int64_t elem_cnt, const int64_t num_rows, const int64_t num_cols,
                        const int64_t diagonal, const T* x, const T fill, T* y) {
  int64_t matrix_size = num_rows * num_cols;
  CUDA_1D_KERNEL_LOOP_T(int64_t, k, elem_cnt) {
    int64_t offset_in_matrix = k % matrix_size;
    int64_t i = offset_in_matrix / num_cols;
    int64_t j = offset_in_matrix - num_cols * i;
    y[k] = j > i + diagonal ? fill : x[k];
  }
}

}  // namespace

template<typename T>
class GpuTrilKernel final : public user_op::OpKernel {
 public:
  GpuTrilKernel() = default;
  ~GpuTrilKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto shape = x->shape();
    const auto diagonal = ctx->Attr<int64_t>("diagonal");
    const int64_t num_rows = shape.At(shape.NumAxes() - 2);
    const int64_t num_cols = shape.At(shape.NumAxes() - 1);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = shape.elem_cnt();
    const T fill = ctx->Attr<bool>("is_floating_fill_value")
                       ? static_cast<T>(ctx->Attr<double>("floating_fill_value"))
                       : static_cast<T>(ctx->Attr<int64_t>("integer_fill_value"));
    RUN_CUDA_KERNEL((TrilGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, num_rows, num_cols,
                    diagonal, x->dptr<T>(), fill, y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_TRIL_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("tril").SetCreateFn<GpuTrilKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "gpu")                                            \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_TRIL_KERNEL(float)
REGISTER_GPU_TRIL_KERNEL(double)
REGISTER_GPU_TRIL_KERNEL(int8_t)
REGISTER_GPU_TRIL_KERNEL(int32_t)
REGISTER_GPU_TRIL_KERNEL(int64_t)
REGISTER_GPU_TRIL_KERNEL(float16)

}  // namespace oneflow
