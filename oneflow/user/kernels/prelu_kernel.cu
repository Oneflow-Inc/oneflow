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

namespace {

template<typename T>
__global__ void PReluForwardGpu(const int64_t elem_cnt, const T* x, const T* alpha, T* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { y[i] = x[i] > 0 ? x[i] : x[i] * alpha[i]; }
}

template<typename T>
__global__ void PReluXBackwardGpu(const int64_t elem_cnt, const T* x, const T* alpha, const T* dy,
                                  T* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { dx[i] = x[i] > 0 ? dy[i] : dy[i] * alpha[i]; }
}

template<typename T>
__global__ void PReluAlphaBackwardGpu(const int64_t elem_cnt, const T* x, const T* dy,
                                      T* alpha_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { alpha_diff[i] = x[i] > 0 ? 0 : dy[i] * x[i]; }
}

}  // namespace

template<typename T>
class GpuPReluKernel final : public user_op::OpKernel {
 public:
  GpuPReluKernel() = default;
  ~GpuPReluKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    user_op::Tensor* broadcasted_alpha = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const Shape& left_extended_shape =
        CreateLeftExtendedShape(ShapeView(alpha->shape()), x->shape().NumAxes());
    NdarrayUtil<DeviceType::kGPU, T>::BroadcastTo(
        ctx->device_ctx(), XpuVarNdarray<T>(x->shape(), broadcasted_alpha->mut_dptr<T>()),
        XpuVarNdarray<const T>(left_extended_shape, alpha->dptr<T>()));

    RUN_CUDA_KERNEL((PReluForwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, x->dptr<T>(),
                    broadcasted_alpha->dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_PRELU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("prelu")                                                       \
      .SetCreateFn<GpuPReluKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                             \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                             \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("x", 0);                   \
        return GetCudaAlignedSize(in_shape->elem_cnt() * sizeof(dtype));              \
      });

REGISTER_GPU_PRELU_KERNEL(float)
REGISTER_GPU_PRELU_KERNEL(double)

template<typename T>
class GpuPReluXGradKernel final : public user_op::OpKernel {
 public:
  GpuPReluXGradKernel() = default;
  ~GpuPReluXGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* broadcasted_alpha = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const Shape& left_extended_shape =
        CreateLeftExtendedShape(ShapeView(alpha->shape()), x->shape().NumAxes());
    NdarrayUtil<DeviceType::kGPU, T>::BroadcastTo(
        ctx->device_ctx(), XpuVarNdarray<T>(x->shape(), broadcasted_alpha->mut_dptr<T>()),
        XpuVarNdarray<const T>(left_extended_shape, alpha->dptr<T>()));

    RUN_CUDA_KERNEL((PReluXBackwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, x->dptr<T>(),
                    broadcasted_alpha->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_PRELU_X_GRAD_KERNEL(dtype)                                        \
  REGISTER_USER_KERNEL("prelu_x_grad")                                                 \
      .SetCreateFn<GpuPReluXGradKernel<dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                              \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("x", 0);                    \
        return GetCudaAlignedSize(in_shape->elem_cnt() * sizeof(dtype));               \
      });

REGISTER_GPU_PRELU_X_GRAD_KERNEL(float)
REGISTER_GPU_PRELU_X_GRAD_KERNEL(double)

template<typename T>
class GpuPReluAlphaGradKernel final : public user_op::OpKernel {
 public:
  GpuPReluAlphaGradKernel() = default;
  ~GpuPReluAlphaGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* alpha_diff = ctx->Tensor4ArgNameAndIndex("alpha_diff", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    T* broadcasted_alpha_diff = tmp_buffer->mut_dptr<T>();
    T* reduce_sum_tmp_buf = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                                 + GetCudaAlignedSize(elem_cnt * sizeof(T)));
    RUN_CUDA_KERNEL((PReluAlphaBackwardGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, x->dptr<T>(),
                    dy->dptr<T>(), broadcasted_alpha_diff);
    const Shape& left_extended_shape =
        CreateLeftExtendedShape(ShapeView(alpha->shape()), x->shape().NumAxes());
    NdarrayUtil<DeviceType::kGPU, T>::ReduceSum(
        ctx->device_ctx(), XpuVarNdarray<T>(left_extended_shape, alpha_diff->mut_dptr<T>()),
        XpuVarNdarray<const T>(x->shape(), broadcasted_alpha_diff),
        XpuVarNdarray<T>(x->shape(), reduce_sum_tmp_buf));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_PRELU_ALPHA_GRAD_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("prelu_alpha_grad")                                                     \
      .SetCreateFn<GpuPReluAlphaGradKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                      \
                       & (user_op::HobDataType("alpha_diff", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                      \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("x", 0);                            \
        return GetCudaAlignedSize(in_shape->elem_cnt() * sizeof(dtype))                        \
               + GetCudaAlignedSize(in_shape->elem_cnt() * sizeof(dtype));                     \
      });

REGISTER_GPU_PRELU_ALPHA_GRAD_KERNEL(float)
REGISTER_GPU_PRELU_ALPHA_GRAD_KERNEL(double)

}  // namespace oneflow
