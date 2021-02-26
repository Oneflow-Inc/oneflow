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
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<typename T>
class CpuPReluKernel final : public user_op::OpKernel {
 public:
  CpuPReluKernel() = default;
  ~CpuPReluKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    user_op::Tensor* broadcasted_alpha = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* x_ptr = x->dptr<T>();
    T* y_ptr = y->mut_dptr<T>();
    T* broadcasted_alpha_ptr = broadcasted_alpha->mut_dptr<T>();
    const int32_t elem_cnt = x->shape().elem_cnt();
    const Shape& left_extended_shape =
        CreateLeftExtendedShape(ShapeView(alpha->shape()), x->shape().NumAxes());
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastTo(
        ctx->device_ctx(), XpuVarNdarray<T>(x->shape(), broadcasted_alpha_ptr),
        XpuVarNdarray<const T>(left_extended_shape, alpha->dptr<T>()));
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      y_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : x_ptr[i] * broadcasted_alpha_ptr[i];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_PRELU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("prelu")                                                       \
      .SetCreateFn<CpuPReluKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                             \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                             \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("x", 0);                   \
        return GetCudaAlignedSize(in_shape->elem_cnt() * sizeof(dtype));              \
      });

REGISTER_CPU_PRELU_KERNEL(float)
REGISTER_CPU_PRELU_KERNEL(double)

template<typename T>
class CpuPReluGradKernel final : public user_op::OpKernel {
 public:
  CpuPReluGradKernel() = default;
  ~CpuPReluGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* alpha_diff = ctx->Tensor4ArgNameAndIndex("alpha_diff", 0);
    const T* x_ptr = x->dptr<T>();
    const T* dy_ptr = dy->dptr<T>();
    T* dx_ptr = dx->mut_dptr<T>();
    const int32_t elem_cnt = x->shape().elem_cnt();
    T* broadcasted_alpha_ptr = tmp_buffer->mut_dptr<T>();
    T* broadcasted_alpha_diff = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                                     + GetCudaAlignedSize(elem_cnt * sizeof(T)));
    T* reduce_sum_tmp_buf = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                                 + 2 * GetCudaAlignedSize(elem_cnt * sizeof(T)));
    const Shape& left_extended_shape =
        CreateLeftExtendedShape(ShapeView(alpha->shape()), x->shape().NumAxes());
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastTo(
        ctx->device_ctx(), XpuVarNdarray<T>(x->shape(), broadcasted_alpha_ptr),
        XpuVarNdarray<const T>(left_extended_shape, alpha->dptr<T>()));
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      dx_ptr[i] = x_ptr[i] > 0 ? dy_ptr[i] : dy_ptr[i] * broadcasted_alpha_ptr[i];
      broadcasted_alpha_diff[i] = x_ptr[i] > 0 ? 0 : dy_ptr[i] * x_ptr[i];
    }
    NdarrayUtil<DeviceType::kCPU, T>::ReduceSum(
        ctx->device_ctx(), XpuVarNdarray<T>(left_extended_shape, alpha_diff->mut_dptr<T>()),
        XpuVarNdarray<const T>(x->shape(), broadcasted_alpha_diff),
        XpuVarNdarray<T>(x->shape(), reduce_sum_tmp_buf));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_PRELU_GRAD_KERNEL(dtype)                                          \
  REGISTER_USER_KERNEL("prelu_grad")                                                   \
      .SetCreateFn<CpuPReluGradKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                              \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("x", 0);                    \
        return 3 * GetCudaAlignedSize(in_shape->elem_cnt() * sizeof(dtype));           \
      });

REGISTER_CPU_PRELU_GRAD_KERNEL(float)
REGISTER_CPU_PRELU_GRAD_KERNEL(double)

}  // namespace oneflow
