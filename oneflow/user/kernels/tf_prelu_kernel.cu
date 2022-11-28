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
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void BroadcastPReluForwardGpu(const int32_t elem_cnt, const int32_t alpha_size,
                                         const int32_t inner_size, const T* x, const T* alpha,
                                         T* y) {
  T zero_val = static_cast<T>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T alpha_i = alpha[(i / inner_size) % alpha_size];
    y[i] = x_i > zero_val ? x_i : x_i * alpha_i;
  }
}

template<typename T>
__global__ void BroadcastPReluBackwardGpu(const int32_t elem_cnt, const int32_t alpha_size,
                                          const int32_t inner_size, const T* x, const T* alpha,
                                          const T* dy, T* dx, T* alpha_diff) {
  T zero_val = static_cast<T>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T dy_i = dy[i];
    const T alpha_i = alpha[(i / inner_size) % alpha_size];
    T dx_i = zero_val;
    T alpha_diff_i = zero_val;
    if (x_i > zero_val) {
      dx_i = dy_i;
      alpha_diff_i = zero_val;
    } else {
      dx_i = dy_i * alpha_i;
      alpha_diff_i = dy_i * x_i;
    }
    dx[i] = dx_i;
    alpha_diff[i] = alpha_diff_i;
  }
}

template<typename T>
__global__ void ElemwisePReluForwardGpu(const int32_t elem_cnt, const T* x, const T* alpha, T* y) {
  T zero_val = static_cast<T>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T alpha_i = alpha[i];
    y[i] = x_i > zero_val ? x_i : x_i * alpha_i;
  }
}

template<typename T>
__global__ void ElemwisePReluBackwardGpu(const int32_t elem_cnt, const T* x, const T* alpha,
                                         const T* dy, T* dx, T* alpha_diff) {
  T zero_val = static_cast<T>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T dy_i = dy[i];
    const T alpha_i = alpha[i];
    T dx_i = zero_val;
    T alpha_diff_i = zero_val;
    if (x_i > zero_val) {
      dx_i = dy_i;
      alpha_diff_i = zero_val;
    } else {
      dx_i = dy_i * alpha_i;
      alpha_diff_i = dy_i * x_i;
    }
    dx[i] = dx_i;
    alpha_diff[i] = alpha_diff_i;
  }
}

bool IsAlphaShapeContiguous(const ShapeView& alpha_shape, const ShapeView& x_shape) {
  if (alpha_shape.elem_cnt() == 1) { return true; }
  int64_t begin_idx = -1;
  for (int64_t i = 0; i < alpha_shape.NumAxes(); ++i) {
    if (alpha_shape.At(i) != 1) {
      begin_idx = i;
      break;
    }
  }
  CHECK_NE(begin_idx, -1);
  int64_t end_idx = -1;
  for (int64_t i = alpha_shape.NumAxes(); i > 0; --i) {
    if (alpha_shape.At(i - 1) != 1) {
      end_idx = i;
      break;
    }
  }
  CHECK_NE(end_idx, -1);
  if (alpha_shape.elem_cnt() == x_shape.Count(begin_idx + 1, end_idx + 1)) {
    return true;
  } else {
    return false;
  }
}

int32_t GetOuterSize(const ShapeView& alpha_shape, const ShapeView& x_shape) {
  int32_t outer_size = x_shape.At(0);
  for (int32_t i = 0; i < alpha_shape.NumAxes(); ++i) {
    if (alpha_shape.At(i) == 1) {
      outer_size *= x_shape.At(i + 1);
    } else {
      break;
    }
  }
  return outer_size;
}

}  // namespace

template<typename T>
class TfGpuPReluKernel final : public user_op::OpKernel {
 public:
  TfGpuPReluKernel() = default;
  ~TfGpuPReluKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape_view().elem_cnt();
    if (IsAlphaShapeContiguous(alpha->shape_view(), x->shape_view())) {
      const int32_t outer_size = GetOuterSize(alpha->shape_view(), x->shape_view());
      const int32_t alpha_size = alpha->shape_view().elem_cnt();
      const int32_t inner_size = elem_cnt / outer_size / alpha_size;
      BroadcastPReluForwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                    ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), y->mut_dptr<T>());
    } else {
      user_op::Tensor* broadcasted_alpha = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      const Shape& left_extended_shape =
          CreateLeftExtendedShape(ShapeView(alpha->shape_view()), x->shape_view().NumAxes());
      NdarrayUtil<DeviceType::kCUDA, T>::BroadcastTo(
          ctx->stream(), XpuVarNdarray<T>(x->shape_view(), broadcasted_alpha->mut_dptr<T>()),
          XpuVarNdarray<const T>(left_extended_shape, alpha->dptr<T>()));
      ElemwisePReluForwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                   ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, x->dptr<T>(), broadcasted_alpha->dptr<T>(), y->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TF_CUDA_PRELU_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("tf_prelu")                                                     \
      .SetCreateFn<TfGpuPReluKernel<dtype>>()                                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                              \
        const Shape& in_shape = ctx->InputShape("x", 0);                               \
        const Shape& alpha_shape = ctx->InputShape("alpha", 0);                        \
        const int64_t tmp_buffer_size =                                                \
            IsAlphaShapeContiguous(alpha_shape, in_shape)                              \
                ? 0                                                                    \
                : GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(dtype));             \
        return tmp_buffer_size;                                                        \
      });

REGISTER_TF_CUDA_PRELU_KERNEL(half)
REGISTER_TF_CUDA_PRELU_KERNEL(float)
REGISTER_TF_CUDA_PRELU_KERNEL(double)

template<typename T>
class TfGpuPReluGradKernel final : public user_op::OpKernel {
 public:
  TfGpuPReluGradKernel() = default;
  ~TfGpuPReluGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* alpha_diff = ctx->Tensor4ArgNameAndIndex("alpha_diff", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t elem_cnt = x->shape_view().elem_cnt();
    T* broadcasted_alpha_diff = tmp_buffer->mut_dptr<T>();
    T* reduce_sum_tmp_buf = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                                 + GetCudaAlignedSize(elem_cnt * sizeof(T)));
    const Shape& left_extended_shape =
        CreateLeftExtendedShape(ShapeView(alpha->shape_view()), x->shape_view().NumAxes());
    if (IsAlphaShapeContiguous(alpha->shape_view(), x->shape_view())) {
      const int32_t outer_size = GetOuterSize(alpha->shape_view(), x->shape_view());
      const int32_t alpha_size = alpha->shape_view().elem_cnt();
      const int32_t inner_size = elem_cnt / outer_size / alpha_size;
      BroadcastPReluBackwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                     ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), dy->dptr<T>(),
          dx->mut_dptr<T>(), broadcasted_alpha_diff);
    } else {
      T* broadcasted_alpha = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                                  + 2 * GetCudaAlignedSize(elem_cnt * sizeof(T)));

      NdarrayUtil<DeviceType::kCUDA, T>::BroadcastTo(
          ctx->stream(), XpuVarNdarray<T>(x->shape_view(), broadcasted_alpha),
          XpuVarNdarray<const T>(left_extended_shape, alpha->dptr<T>()));

      ElemwisePReluBackwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                    ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, x->dptr<T>(), broadcasted_alpha, dy->dptr<T>(), dx->mut_dptr<T>(),
          broadcasted_alpha_diff);
    }
    NdarrayUtil<DeviceType::kCUDA, T>::ReduceSum(
        ctx->stream(), XpuVarNdarray<T>(left_extended_shape, alpha_diff->mut_dptr<T>()),
        XpuVarNdarray<const T>(x->shape_view(), broadcasted_alpha_diff),
        XpuVarNdarray<T>(x->shape_view(), reduce_sum_tmp_buf));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TF_CUDA_PRELU_GRAD_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("tf_prelu_grad")                                                 \
      .SetCreateFn<TfGpuPReluGradKernel<dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
        const Shape& in_shape = ctx->InputShape("x", 0);                                \
        const Shape& alpha_shape = ctx->InputShape("alpha", 0);                         \
        const int64_t tmp_buffer_size =                                                 \
            IsAlphaShapeContiguous(alpha_shape, in_shape)                               \
                ? 2 * GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(dtype))           \
                : 3 * GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(dtype));          \
        return tmp_buffer_size;                                                         \
      });

REGISTER_TF_CUDA_PRELU_GRAD_KERNEL(half)
REGISTER_TF_CUDA_PRELU_GRAD_KERNEL(float)
REGISTER_TF_CUDA_PRELU_GRAD_KERNEL(double)

}  // namespace oneflow
