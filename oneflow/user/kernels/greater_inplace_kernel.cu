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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GreaterInplaceForward(const int32_t n, const T* x, const T* y, T* out) {
  CUDA_1D_KERNEL_LOOP_T(int32_t, i, n) {
    out[i] = x[i] > y[i] ? static_cast<T>(1) : static_cast<T>(0);
  }
}

}  // namespace

template<typename T>
class GpuGreaterInplaceKernel final : public user_op::OpKernel {
 public:
  GpuGreaterInplaceKernel() = default;
  ~GpuGreaterInplaceKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // TODO
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* broadcast_y_ptr = tmp_buffer->mut_dptr<T>();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    int64_t like_ndim = x->shape_view().NumAxes();
    int64_t x_ndim = y->shape_view().NumAxes();
    int64_t num_prepend = like_ndim - x_ndim;
    std::vector<int64_t> prepend_shape(num_prepend, 1);
    std::vector<int32_t> broadcast_axes;
    for (int i = 0; i < x_ndim; ++i) { prepend_shape.emplace_back(y->shape_view().At(i)); }
    for (int i = 0; i < num_prepend; ++i) { broadcast_axes.emplace_back(i); }
    for (int i = num_prepend; i < prepend_shape.size(); ++i) {
      if (prepend_shape[i] != x->shape_view().At(i)) {
        if (prepend_shape[i] == 1) { broadcast_axes.emplace_back(i); }
      }
    }
    const Shape& reduced_shape = CreateReducedShapeOrOnesShape(
        x->shape_view(), {broadcast_axes.begin(), broadcast_axes.end()});

    const int32_t elem_cnt = x->shape_view().elem_cnt();
    NdarrayUtil<DeviceType::kCUDA, T>::BroadcastTo(
        ctx->stream(), XpuVarNdarray<T>(x->shape_view(), broadcast_y_ptr),
        XpuVarNdarray<const T>(reduced_shape, y->dptr<T>()));

    RUN_CUDA_KERNEL((GreaterInplaceForward<T>), ctx->stream(), elem_cnt, elem_cnt, x->dptr<T>(),
                    broadcast_y_ptr, out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_GREATER_INPLACE_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("broadcast_inplace_greater")                                      \
      .SetCreateFn<GpuGreaterInplaceKernel<dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                   \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                       \
        const Shape& x_shape = ctx->InputShape("x", 0);                                  \
        return GetCudaAlignedSize(x_shape.elem_cnt() * sizeof(dtype));                   \
      });

REGISTER_CUDA_GREATER_INPLACE_KERNEL(half)
REGISTER_CUDA_GREATER_INPLACE_KERNEL(float)
REGISTER_CUDA_GREATER_INPLACE_KERNEL(double)
// REGISTER_CUDA_GREATER_INPLACE_KERNEL(uint8_t)
REGISTER_CUDA_GREATER_INPLACE_KERNEL(int8_t)
REGISTER_CUDA_GREATER_INPLACE_KERNEL(int32_t)
REGISTER_CUDA_GREATER_INPLACE_KERNEL(int64_t)

}  // namespace oneflow
