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
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace {

Shape CreatePreluLeftExtendedShape(const ShapeView& shape) {
  DimVector dim_vec(shape.NumAxes());
  const size_t left_ones_num = 1;
  int i = 0;
  for (; i < left_ones_num; ++i) { dim_vec.at(i) = 1LL; }
  for (; i < shape.NumAxes(); ++i) { dim_vec.at(i) = shape.At(i); }
  return Shape(std::move(dim_vec));
}

template<typename T>
__global__ void BroadcastPReluSingleAlphaForwardGpu(const int32_t elem_cnt,
                                                    const int32_t alpha_size,
                                                    const int32_t inner_size, const T* x,
                                                    const T* alpha, T* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    y[i] = x_i > 0 ? x_i : x_i * alpha[0];
  }
}

template<typename T>
__global__ void BroadcastPReluSingleAlphaBackwardGpu(const int32_t elem_cnt,
                                                     const int32_t alpha_size,
                                                     const int32_t inner_size, const T* x,
                                                     const T* alpha, const T* dy, T* dx,
                                                     T* alpha_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T dy_i = dy[i];
    T dx_i = 0;
    T alpha_diff_i = 0;
    if (x_i > 0) {
      dx_i = dy_i;
      alpha_diff_i = 0;
    } else {
      dx_i = dy_i * alpha[0];
      alpha_diff_i = dy_i * x_i;
    }
    dx[i] = dx_i;
    alpha_diff[i] = alpha_diff_i;
  }
}

template<>
__global__ void BroadcastPReluSingleAlphaForwardGpu<half>(const int32_t elem_cnt,
                                                          const int32_t alpha_size,
                                                          const int32_t inner_size, const half* x,
                                                          const half* alpha, half* y) {
  half zero = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half x_i = x[i];
    y[i] = x_i > zero ? x_i : __hmul(x_i, alpha[0]);
  }
}

template<>
__global__ void BroadcastPReluSingleAlphaBackwardGpu<half>(const int32_t elem_cnt,
                                                           const int32_t alpha_size,
                                                           const int32_t inner_size, const half* x,
                                                           const half* alpha, const half* dy,
                                                           half* dx, half* alpha_diff) {
  half zero = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half x_i = x[i];
    const half dy_i = dy[i];
    half dx_i = 0;
    half alpha_diff_i = 0;
    if (x_i > zero) {
      dx_i = dy_i;
      alpha_diff_i = 0;
    } else {
      dx_i = __hmul(dy_i, alpha[0]);
      alpha_diff_i = __hmul(dy_i, x_i);
    }
    dx[i] = dx_i;
    alpha_diff[i] = alpha_diff_i;
  }
}

template<typename T>
__global__ void BroadcastPReluMultiAlphaForwardGpu(const int32_t elem_cnt, const int32_t alpha_size,
                                                   const int32_t inner_size, const T* x,
                                                   const T* alpha, T* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    int32_t i_div_inner_size = i / inner_size;
    int32_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int32_t alpha_i = i_div_inner_size - idx_sub_alpha;
    y[i] = x_i > 0 ? x_i : x_i * alpha[alpha_i];
  }
}

template<typename T>
__global__ void BroadcastPReluMultiAlphaBackwardGpu(const int32_t elem_cnt,
                                                    const int32_t alpha_size,
                                                    const int32_t inner_size, const T* x,
                                                    const T* alpha, const T* dy, T* dx,
                                                    T* alpha_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T dy_i = dy[i];
    int32_t i_div_inner_size = i / inner_size;
    int32_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int32_t alpha_i = i_div_inner_size - idx_sub_alpha;
    T dx_i = 0;
    T alpha_diff_i = 0;
    if (x_i > 0) {
      dx_i = dy_i;
      alpha_diff_i = 0;
    } else {
      dx_i = dy_i * alpha[alpha_i];
      alpha_diff_i = dy_i * x_i;
    }
    dx[i] = dx_i;
    alpha_diff[i] = alpha_diff_i;
  }
}

template<>
__global__ void BroadcastPReluMultiAlphaForwardGpu<half>(const int32_t elem_cnt,
                                                         const int32_t alpha_size,
                                                         const int32_t inner_size, const half* x,
                                                         const half* alpha, half* y) {
  half zero = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half x_i = x[i];
    int32_t i_div_inner_size = i / inner_size;
    int32_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int32_t alpha_i = i_div_inner_size - idx_sub_alpha;
    y[i] = x_i > zero ? x_i : __hmul(x_i, alpha[alpha_i]);
  }
}

template<>
__global__ void BroadcastPReluMultiAlphaBackwardGpu<half>(const int32_t elem_cnt,
                                                          const int32_t alpha_size,
                                                          const int32_t inner_size, const half* x,
                                                          const half* alpha, const half* dy,
                                                          half* dx, half* alpha_diff) {
  half zero = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half x_i = x[i];
    const half dy_i = dy[i];
    int32_t i_div_inner_size = i / inner_size;
    int32_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int32_t alpha_i = i_div_inner_size - idx_sub_alpha;
    half dx_i = 0;
    half alpha_diff_i = 0;
    if (x_i > zero) {
      dx_i = dy_i;
      alpha_diff_i = 0;
    } else {
      dx_i = __hmul(dy_i, alpha[alpha_i]);
      alpha_diff_i = __hmul(dy_i, x_i);
    }
    dx[i] = dx_i;
    alpha_diff[i] = alpha_diff_i;
  }
}

}  // namespace

template<typename T>
class GpuPReluKernel final : public user_op::OpKernel {
 public:
  GpuPReluKernel() = default;
  ~GpuPReluKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const int32_t batch = x->shape().At(0);
    const int32_t channels = x->shape().At(1);
    const int32_t alpha_size = alpha->shape().elem_cnt();
    const int32_t inner_size = elem_cnt / batch / channels;
    int grid_size;
    cudaError_t err = cuda::elementwise::GetNumBlocks(1, &grid_size);

    if (alpha_size == 1) {
      BroadcastPReluSingleAlphaForwardGpu<T>
          <<<grid_size, cuda::elementwise::kBlockSize, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), y->mut_dptr<T>());
    } else {
      BroadcastPReluMultiAlphaForwardGpu<T><<<grid_size, cuda::elementwise::kBlockSize, 0,
                                              ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), y->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_PRELU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("prelu").SetCreateFn<GpuPReluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                 \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_PRELU_KERNEL(half)
REGISTER_CUDA_PRELU_KERNEL(float)
REGISTER_CUDA_PRELU_KERNEL(double)

template<typename T>
class GpuPReluGradKernel final : public user_op::OpKernel {
 public:
  GpuPReluGradKernel() = default;
  ~GpuPReluGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* alpha_diff = ctx->Tensor4ArgNameAndIndex("alpha_diff", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    int grid_size;
    cudaError_t err = cuda::elementwise::GetNumBlocks(1, &grid_size);

    T* broadcasted_alpha_diff = tmp_buffer->mut_dptr<T>();
    T* reduce_sum_tmp_buf = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                                 + GetCudaAlignedSize(elem_cnt * sizeof(T)));

    const Shape& left_extended_shape = CreatePreluLeftExtendedShape(ShapeView(x->shape()));

    const int32_t batch = x->shape().At(0);
    const int32_t channels = x->shape().At(1);
    const int32_t alpha_size = alpha->shape().elem_cnt();
    const int32_t inner_size = elem_cnt / batch / channels;
    if (alpha_size == 1) {
      BroadcastPReluSingleAlphaBackwardGpu<T>
          <<<grid_size, cuda::elementwise::kBlockSize, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), dy->dptr<T>(),
              dx->mut_dptr<T>(), broadcasted_alpha_diff);
    } else {
      BroadcastPReluMultiAlphaBackwardGpu<T>
          <<<grid_size, cuda::elementwise::kBlockSize, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), dy->dptr<T>(),
              dx->mut_dptr<T>(), broadcasted_alpha_diff);
    }

    NdarrayUtil<DeviceType::kCUDA, T>::ReduceSum(
        ctx->stream(), XpuVarNdarray<T>(left_extended_shape, alpha_diff->mut_dptr<T>()),
        XpuVarNdarray<const T>(x->shape(), broadcasted_alpha_diff),
        XpuVarNdarray<T>(x->shape(), reduce_sum_tmp_buf));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_PRELU_GRAD_KERNEL(dtype)                                          \
  REGISTER_USER_KERNEL("prelu_grad")                                                    \
      .SetCreateFn<GpuPReluGradKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
        const Shape& in_shape = ctx->InputShape("x", 0);                                \
        const Shape& alpha_shape = ctx->InputShape("alpha", 0);                         \
        const int64_t tmp_buffer_size =                                                 \
            2 * GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(dtype));                \
        return tmp_buffer_size;                                                         \
      });

REGISTER_CUDA_PRELU_GRAD_KERNEL(half)
REGISTER_CUDA_PRELU_GRAD_KERNEL(float)
REGISTER_CUDA_PRELU_GRAD_KERNEL(double)

}  // namespace oneflow
