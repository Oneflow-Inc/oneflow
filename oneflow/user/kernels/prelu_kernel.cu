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

constexpr int32_t kBlockSize = 256;

template<typename T, typename IndexType, int pack_size>
__global__ void PReluForwardMultiAlphaGpu(const IndexType elem_cnt, const IndexType alpha_size,
                                          const IndexType inner_size, const IndexType mul_size,
                                          const T* x, const T* alpha, T* y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  T zero_val = static_cast<T>(0);
  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    IndexType idx_sub_alpha = linear_index / mul_size * alpha_size;
    IndexType alpha_idx = linear_index / inner_size - idx_sub_alpha;

    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    LoadPack y_vec;
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      y_vec.elem[i] = x_vec.elem[i] > zero_val ? x_vec.elem[i] : x_vec.elem[i] * alpha[alpha_idx];
    }
    *(reinterpret_cast<LoadType*>(y + linear_index)) = y_vec.storage;
  }
}

template<typename T, typename IndexType, int pack_size>
__global__ void PReluBackwardMultiAlphaGpu(const IndexType elem_cnt, const IndexType alpha_size,
                                           const IndexType inner_size, const IndexType mul_size,
                                           const T* x, const T* alpha, const T* dy, T* dx,
                                           T* alpha_diff) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  T zero_val = static_cast<T>(0);

  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    IndexType idx_sub_alpha = linear_index / mul_size * alpha_size;
    IndexType alpha_idx = linear_index / inner_size - idx_sub_alpha;

    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    const LoadType* dy_load = reinterpret_cast<const LoadType*>(dy + linear_index);
    LoadPack dy_vec;
    dy_vec.storage = *dy_load;

    LoadPack dx_vec;
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      dx_vec.elem[i] =
          x_vec.elem[i] > zero_val ? dy_vec.elem[i] : dy_vec.elem[i] * alpha[alpha_idx];
      alpha_diff[alpha_idx] += x_vec.elem[i] > zero_val ? zero_val : dy_vec.elem[i] * x_vec.elem[i];
    }

    *(reinterpret_cast<LoadType*>(dx + linear_index)) = dx_vec.storage;
  }
}

template<typename T>
void DispatchPreluForwardIndex(ep::Stream* stream, const int64_t elem_cnt, const int64_t alpha_size,
                               const int64_t inner_size, const T* x, const T* alpha, T* y) {
  constexpr int pack_size = cuda::elementwise::PackSize<T>();
  const int64_t pack_num = elem_cnt / pack_size;
  int grid_size;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);

  const int64_t alpha_inner_size = alpha_size * inner_size;

  if (elem_cnt < GetMaxVal<int32_t>()) {
    PReluForwardMultiAlphaGpu<T, int32_t, pack_size>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, alpha_inner_size, x, alpha, y);
  } else {
    PReluForwardMultiAlphaGpu<T, int64_t, pack_size>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, alpha_inner_size, x, alpha, y);
  }
}

template<typename T>
void DispatchPreluBackwardIndex(ep::Stream* stream, const int64_t elem_cnt,
                                const int64_t alpha_size, const int64_t inner_size, const T* x,
                                const T* alpha, const T* dy, T* dx, T* alpha_diff) {
  constexpr int pack_size = cuda::elementwise::PackSize<T>();
  const int64_t pack_num = elem_cnt / pack_size;
  int grid_size;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);

  const int64_t alpha_inner_size = alpha_size * inner_size;

  if (elem_cnt < GetMaxVal<int32_t>()) {
    PReluBackwardMultiAlphaGpu<T, int32_t, pack_size>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, alpha_inner_size, x, alpha, dy, dx, alpha_diff);
  } else {
    PReluBackwardMultiAlphaGpu<T, int64_t, pack_size>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, alpha_inner_size, x, alpha, dy, dx, alpha_diff);
  }
}

template<typename T>
__global__ void PReluForwardNaiveMultiAlphaGpu(const int64_t elem_cnt, const int64_t alpha_size,
                                               const int64_t inner_size, const T* x, const T* alpha,
                                               T* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t i_div_inner_size = i / inner_size;
    int64_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int64_t alpha_idx = i_div_inner_size - idx_sub_alpha;
    const T x_i = x[i];
    const T alpha_i = alpha[alpha_idx];
    y[i] = x_i > 0 ? x_i : x_i * alpha_i;
  }
}

template<typename T>
__global__ void PReluBackwardNaiveMultiAlphaGpu(const int64_t elem_cnt, const int64_t alpha_size,
                                                const int64_t inner_size, const T* x,
                                                const T* alpha, const T* dy, T* dx, T* alpha_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t i_div_inner_size = i / inner_size;
    int64_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int64_t alpha_idx = i_div_inner_size - idx_sub_alpha;
    const T x_i = x[i];
    const T dy_i = dy[i];
    const T alpha_i = alpha[alpha_idx];
    dx[i] = x_i > 0 ? dy_i : dy_i * alpha_i;
    alpha_diff[alpha_idx] += x_i > 0 ? 0 : dy_i * x_i;
  }
}

template<>
__global__ void PReluForwardNaiveMultiAlphaGpu<half>(const int64_t elem_cnt,
                                                     const int64_t alpha_size,
                                                     const int64_t inner_size, const half* x,
                                                     const half* alpha, half* y) {
  half zero_val = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t i_div_inner_size = i / inner_size;
    int64_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int64_t alpha_idx = i_div_inner_size - idx_sub_alpha;
    const half x_i = x[i];
    const half alpha_i = alpha[alpha_idx];
    y[i] = x_i > zero_val ? x_i : __hmul(x_i, alpha_i);
  }
}

template<>
__global__ void PReluBackwardNaiveMultiAlphaGpu<half>(const int64_t elem_cnt,
                                                      const int64_t alpha_size,
                                                      const int64_t inner_size, const half* x,
                                                      const half* alpha, const half* dy, half* dx,
                                                      half* alpha_diff) {
  half zero_val = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t i_div_inner_size = i / inner_size;
    int64_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int64_t alpha_idx = i_div_inner_size - idx_sub_alpha;
    const half x_i = x[i];
    const half dy_i = dy[i];
    const half alpha_i = alpha[alpha_idx];
    dx[i] = x_i > zero_val ? dy_i : __hmul(dy_i, alpha_i);
    alpha_diff[alpha_idx] =
        __hadd(alpha_diff[alpha_idx], x_i > zero_val ? zero_val : __hmul(dy_i, x_i));
  }
}

template<typename T>
__global__ void PReluForwardSingleAlphaGpu(const int64_t elem_cnt, const T* x, const T* alpha,
                                           T* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { y[i] = x[i] > 0 ? x[i] : x[i] * alpha[0]; }
}

template<typename T>
__global__ void PReluBackwardSingleAlphaGpu(const int64_t elem_cnt, const T* x, const T* alpha,
                                            const T* dy, T* dx, T* alpha_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    dx[i] = x[i] > 0 ? dy[i] : dy[i] * alpha[0];
    alpha_diff[0] += x[i] > 0 ? 0 : dy[i] * x[i];
  }
}

template<>
__global__ void PReluForwardSingleAlphaGpu<half>(const int64_t elem_cnt, const half* x,
                                                 const half* alpha, half* y) {
  const half zero_val = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { y[i] = x[i] > zero_val ? x[i] : __hmul(x[i], alpha[0]); }
}

template<>
__global__ void PReluBackwardSingleAlphaGpu<half>(const int64_t elem_cnt, const half* x,
                                                  const half* alpha, const half* dy, half* dx,
                                                  half* alpha_diff) {
  const half zero_val = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    dx[i] = x[i] > zero_val ? dy[i] : __hmul(dy[i], alpha[0]);
    alpha_diff[0] = __hadd(alpha_diff[0], x[i] > zero_val ? zero_val : __hmul(dy[i], x[i]));
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

    const int64_t elem_cnt = x->shape().elem_cnt();
    const int64_t batch = x->shape().At(0);
    const int64_t channels = x->shape().At(1);
    const int64_t inner_size = elem_cnt / batch / channels;
    const int64_t alpha_size = alpha->shape().elem_cnt();
    constexpr int pack_size = cuda::elementwise::PackSize<T>();
    int grid_size;
    cudaError_t err = cuda::elementwise::GetNumBlocks(1, &grid_size);

    if (alpha_size == 1) {
      PReluForwardSingleAlphaGpu<T>
          <<<grid_size, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, x->dptr<T>(), alpha->dptr<T>(), y->mut_dptr<T>());
    } else if (inner_size % alpha_size != 0) {
      PReluForwardNaiveMultiAlphaGpu<T>
          <<<grid_size, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), y->mut_dptr<T>());
    } else {
      DispatchPreluForwardIndex<T>(
          ctx->stream(), elem_cnt, alpha_size, inner_size, reinterpret_cast<const T*>(x->dptr()),
          reinterpret_cast<const T*>(alpha->dptr()), reinterpret_cast<T*>(y->mut_dptr()));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_PRELU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("prelu").SetCreateFn<GpuPReluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                 \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_PRELU_KERNEL(float)
REGISTER_CUDA_PRELU_KERNEL(double)
REGISTER_CUDA_PRELU_KERNEL(half)

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

    Memset<DeviceType::kCUDA>(ctx->stream(), alpha_diff->mut_dptr<T>(), 0,
                              alpha_diff->shape().elem_cnt() * sizeof(T));

    const int64_t elem_cnt = x->shape().elem_cnt();
    const int64_t batch = x->shape().At(0);
    const int64_t channels = x->shape().At(1);
    const int64_t inner_size = elem_cnt / batch / channels;
    const int64_t alpha_size = alpha->shape().elem_cnt();
    constexpr int pack_size = cuda::elementwise::PackSize<T>();
    int grid_size;
    cudaError_t err = cuda::elementwise::GetNumBlocks(1, &grid_size);

    if (alpha_size == 1) {
      PReluBackwardSingleAlphaGpu<T>
          <<<grid_size, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, x->dptr<T>(), alpha->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>(),
              alpha_diff->mut_dptr<T>());
    } else if (inner_size % pack_size != 0) {
      PReluBackwardNaiveMultiAlphaGpu<T>
          <<<grid_size, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), dy->dptr<T>(),
              dx->mut_dptr<T>(), alpha_diff->mut_dptr<T>());
    } else {
      DispatchPreluBackwardIndex<T>(
          ctx->stream(), x->shape().elem_cnt(), alpha_size, inner_size,
          reinterpret_cast<const T*>(x->dptr()), reinterpret_cast<const T*>(alpha->dptr()),
          reinterpret_cast<const T*>(dy->dptr()), reinterpret_cast<T*>(dx->mut_dptr()),
          reinterpret_cast<T*>(alpha_diff->mut_dptr()));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_PRELU_GRAD_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("prelu_grad")                                   \
      .SetCreateFn<GpuPReluGradKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_PRELU_GRAD_KERNEL(float)
REGISTER_CUDA_PRELU_GRAD_KERNEL(double)
REGISTER_CUDA_PRELU_GRAD_KERNEL(half)

}  // namespace oneflow
