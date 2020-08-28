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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void ReplaceNonZeroWithOne(const int64_t elem_cnt, T* y) {
  T zero = GetZeroVal<T>();
  T one = GetOneVal<T>();
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, elem_cnt) { y[i] = (y[i] != zero) * one; }
}

}  // namespace

template<typename T>
class GpuMaskedFillKernel final : public user_op::OpKernel {
 public:
  GpuMaskedFillKernel() = default;
  ~GpuMaskedFillKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp_mask_and_y = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T value = static_cast<T>(ctx->Attr<float>("value"));
    const auto mask_shape = mask->shape();
    const auto y_shape = y->shape();
    size_t num_axes = y_shape.NumAxes();
    KernelUtil<DeviceType::kGPU, T>::AddByScalar(ctx->device_ctx(), mask_shape.elem_cnt(),
                                                 mask->dptr<T>(), static_cast<T>(-1),
                                                 tmp_mask_and_y->mut_dptr<T>());
    KernelUtil<DeviceType::kGPU, T>::MulByScalarPara(ctx->device_ctx(), mask_shape.elem_cnt(),
                                                     tmp_mask_and_y->dptr<T>(), static_cast<T>(-1),
                                                     tmp_mask_and_y->mut_dptr<T>());
    NdarrayUtil<DeviceType::kGPU, T>::BroadcastMul(
        ctx->device_ctx(), XpuVarNdarray<T>(y->shape(), y->mut_dptr<T>(), num_axes),
        XpuVarNdarray<const T>(x->shape(), x->dptr<T>(), num_axes),
        XpuVarNdarray<const T>(mask->shape(), tmp_mask_and_y->dptr<T>(), num_axes));
    NdarrayUtil<DeviceType::kGPU, T>::BroadcastMul(
        ctx->device_ctx(),
        XpuVarNdarray<T>(y->shape(), tmp_mask_and_y->mut_dptr<T>() + mask_shape.elem_cnt(),
                         num_axes),
        XpuVarNdarray<const T>(x->shape(), x->dptr<T>(), num_axes),
        XpuVarNdarray<const T>(mask->shape(), mask->dptr<T>(), num_axes));
    T zero = GetZeroVal<T>();
    T* tmp_y_dptr = tmp_mask_and_y->mut_dptr<T>() + mask_shape.elem_cnt();
    RUN_CUDA_KERNEL((ReplaceNonZeroWithOne<T>), ctx->device_ctx(), y->shape().elem_cnt(),
                    y->shape().elem_cnt(), tmp_y_dptr);
    KernelUtil<DeviceType::kGPU, T>::Axpy(ctx->device_ctx(), y->shape().elem_cnt(), value,
                                          tmp_mask_and_y->dptr<T>() + mask_shape.elem_cnt(), 1,
                                          y->mut_dptr<T>(), 1);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_MASKED_FILL_KERNEL(dtype)                                                     \
  REGISTER_USER_KERNEL("masked_fill")                                                              \
      .SetCreateFn<GpuMaskedFillKernel<dtype>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                          \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value))              \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                          \
        const Shape* mask_shape = ctx->Shape4ArgNameAndIndex("mask", 0);                           \
        const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);                                 \
        return GetCudaAlignedSize((mask_shape->elem_cnt() + y_shape->elem_cnt()) * sizeof(dtype)); \
      });

REGISTER_GPU_MASKED_FILL_KERNEL(float)
REGISTER_GPU_MASKED_FILL_KERNEL(double)
REGISTER_GPU_MASKED_FILL_KERNEL(int8_t)
REGISTER_GPU_MASKED_FILL_KERNEL(int32_t)
REGISTER_GPU_MASKED_FILL_KERNEL(int64_t)

template<typename T>
class GpuMaskedFillGradKernel final : public user_op::OpKernel {
 public:
  GpuMaskedFillGradKernel() = default;
  ~GpuMaskedFillGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* tmp_mask_and_y = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto mask_shape = mask->shape();
    const auto dy_shape = dy->shape();
    size_t num_axes = dy_shape.NumAxes();
    T* reduce_sum_tmp_buf =
        tmp_mask_and_y->mut_dptr<T>() + mask_shape.elem_cnt() + dy_shape.elem_cnt();
    KernelUtil<DeviceType::kGPU, T>::AddByScalar(ctx->device_ctx(), mask_shape.elem_cnt(),
                                                 mask->dptr<T>(), static_cast<T>(-1),
                                                 tmp_mask_and_y->mut_dptr<T>());
    KernelUtil<DeviceType::kGPU, T>::MulByScalarPara(ctx->device_ctx(), mask_shape.elem_cnt(),
                                                     tmp_mask_and_y->dptr<T>(), static_cast<T>(-1),
                                                     tmp_mask_and_y->mut_dptr<T>());
    NdarrayUtil<DeviceType::kGPU, T>::BroadcastMul(
        ctx->device_ctx(),
        XpuVarNdarray<T>(dy_shape, tmp_mask_and_y->mut_dptr<T>() + mask_shape.elem_cnt(), num_axes),
        XpuVarNdarray<const T>(dy_shape, dy->dptr<T>(), num_axes),
        XpuVarNdarray<const T>(mask_shape, tmp_mask_and_y->dptr<T>(), num_axes));
    const Shape& left_extend_shape = CreateLeftExtendedShape(dx->shape(), dy_shape.NumAxes());
    NdarrayUtil<DeviceType::kGPU, T>::ReduceSum(
        ctx->device_ctx(), XpuVarNdarray<T>(left_extend_shape, dx->mut_dptr<T>()),
        XpuVarNdarray<const T>(dy_shape, tmp_mask_and_y->mut_dptr<T>() + mask_shape.elem_cnt()),
        XpuVarNdarray<T>(dy_shape, reduce_sum_tmp_buf));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_MASKED_FILL_GRAD_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("masked_fill_grad")                                             \
      .SetCreateFn<GpuMaskedFillGradKernel<dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                              \
        const Shape* mask_shape = ctx->Shape4ArgNameAndIndex("mask", 0);               \
        const Shape* y_shape = ctx->Shape4ArgNameAndIndex("dy", 0);                    \
        return GetCudaAlignedSize(mask_shape->elem_cnt() * sizeof(dtype))              \
               + GetCudaAlignedSize(y_shape->elem_cnt() * sizeof(dtype))               \
               + GetCudaAlignedSize(y_shape->elem_cnt() * sizeof(dtype));              \
      });

REGISTER_GPU_MASKED_FILL_GRAD_KERNEL(float)
REGISTER_GPU_MASKED_FILL_GRAD_KERNEL(double)
REGISTER_GPU_MASKED_FILL_GRAD_KERNEL(int8_t)
REGISTER_GPU_MASKED_FILL_GRAD_KERNEL(int32_t)
REGISTER_GPU_MASKED_FILL_GRAD_KERNEL(int64_t)

}  // namespace oneflow