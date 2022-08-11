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
#include "oneflow/core/common/eigen_util.h"

namespace oneflow {

namespace {

static inline size_t BatchCount(const user_op::Tensor* batched_matrices) {
  size_t result = 1;
  for (size_t i = 0; i < batched_matrices->shape_view().NumAxes() - 2; i++) {
    result *= batched_matrices->shape_view().At(i);
  }
  return result;
}

static inline size_t MatrixStride(const user_op::Tensor* batched_matrices) {
  const int64_t num_axes = batched_matrices->shape_view().NumAxes();
  return batched_matrices->shape_view().At(num_axes - 2)
         * batched_matrices->shape_view().At(num_axes - 1);
}

}  // namespace

template<typename T>
class CpuInvKernel final : public user_op::OpKernel {
 public:
  CpuInvKernel() = default;
  ~CpuInvKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto batch_count = BatchCount(x);
    auto matrix_stride = MatrixStride(x);
    auto matrix_size = x->shape_view().At(x->shape_view().NumAxes() - 2);
    const T* x_ptr = x->dptr<T>();
    T* y_ptr = y->mut_dptr<T>();
    FOR_RANGE(int64_t, i, 0, batch_count) {
      ConstEigenMatrixMap<T> x_mat(x_ptr + i * matrix_stride, matrix_size, matrix_size);
      EigenMatrixMap<T> y_mat(y_ptr + i * matrix_stride, matrix_size, matrix_size);
      if (x_mat.determinant() == 0) {
        LOG(FATAL)
            << "(Batch element " << i
            << "): the inversion could not be completed because the input matrix is singular.";
      }
      y_mat = x_mat.inverse();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_INV_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("inv").SetCreateFn<CpuInvKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                              \
      && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_CPU_INV_KERNEL(float)
REGISTER_CPU_INV_KERNEL(double)

}  // namespace oneflow
