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

constexpr int32_t kVecSize = 4;
constexpr int32_t kBlockSize = 256;

template<typename T>
constexpr int32_t GetPreluPackSize() {
  // For float, bfloat16, half.
  return 4;
};

template<>
constexpr int32_t GetPreluPackSize<half2>() {
  return 2;
};

template<>
constexpr int32_t GetPreluPackSize<double>() {
  return 2;
};

union RandPack4 {
  float4 storage;
  float elem[4];
};

template<typename T>
struct GetPack2Type {
  using T2 = typename std::aligned_storage<2 * sizeof(T), 2 * sizeof(T)>::type;
};

template<>
struct GetPack2Type<half> {
  using T2 = half2;
};

#if CUDA_VERSION >= 11000
template<>
struct GetPack2Type<nv_bfloat16> {
  using T2 = nv_bfloat162;
};
#endif

template<typename T>
using Pack2Type = typename GetPack2Type<T>::T2;

using H2PackType = typename std::aligned_storage<4 * sizeof(half), 4 * sizeof(half)>::type;

template<typename T>
union H2Pack {
  cuda::elementwise::Pack<T, 4> pack_storage;
  Pack2Type<T> h2[2];
  __device__ H2Pack() {
    // do nothing
  }
};

template<>
union H2Pack<half> {
  cuda::elementwise::Pack<half, 4> pack_storage;
  half2 h2[2];
  __device__ H2Pack() {
    // do nothing
  }
};

#if CUDA_VERSION >= 11000
template<>
union H2Pack<nv_bfloat16> {
  cuda::elementwise::Pack<nv_bfloat16, 4> pack_storage;
  nv_bfloat162 h2[2];
  __device__ H2Pack() {
    // do nothing
  }
};
#endif

template<typename T>
__device__ Pack2Type<T> Make2(float v);

template<>
__device__ Pack2Type<half> Make2<half>(float v) {
  return __float2half2_rn(v);
}

#if CUDA_VERSION >= 11000
template<>
__device__ Pack2Type<nv_bfloat16> Make2<nv_bfloat16>(float v) {
  return __float2bfloat162_rn(v);
}
#endif

#if CUDA_VERSION >= 11000
#define RETURN_VOID_IF_HALF                                                                        \
  typename std::enable_if_t<(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value), \
                            void>
#else
#define RETURN_VOID_IF_HALF typename std::enable_if_t<std::is_same<T, half>::value, void>
#endif
#define RETURN_VOID_IF_FLOAT typename std::enable_if_t<std::is_same<T, float>::value, void>
#define RETURN_VOID_IF_DOUBLE typename std::enable_if_t<std::is_same<T, double>::value, void>

template<typename T, int pack_size, bool tail>
__global__ RETURN_VOID_IF_HALF PReluForwardMultiAlphaGpu(
    const int32_t elem_cnt, const int32_t alpha_size, const int32_t inner_size,
    const int64_t n_tail, const T* x, const T* alpha, T* y, const T* tail_x, T* tail_y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t linear_index = global_thread_id; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x) {
    const T* x_load = x + linear_index;
    T* y_load = y + linear_index;
    y_load[0] = x_load[0] > static_cast<T>(0.0)
                    ? x_load[0]
                    : __hmul(x_load[0], alpha[(linear_index / inner_size) % alpha_size]);
  }

  if (tail && global_thread_id < n_tail) {
    const T* x_load = x + global_thread_id;
    T* y_load = y + global_thread_id;
    y_load[0] = x_load[0] > static_cast<T>(0.0)
                    ? x_load[0]
                    : __hmul(x_load[0], alpha[(global_thread_id / inner_size) % alpha_size]);
  }
}

template<typename T, int pack_size, bool tail>
__global__ RETURN_VOID_IF_DOUBLE PReluForwardMultiAlphaGpu(
    const int32_t elem_cnt, const int32_t alpha_size, const int32_t inner_size,
    const int64_t n_tail, const T* x, const T* alpha, T* y, const T* tail_x, T* tail_y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;

  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    LoadPack y_vec;
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      y_vec.elem[i] = x_vec.elem[i] > 0
                          ? x_vec.elem[i]
                          : x_vec.elem[i] * alpha[(linear_index / inner_size) % alpha_size];
    }
    *(reinterpret_cast<LoadType*>(y + linear_index)) = y_vec.storage;
  }

  if (tail && global_thread_id < n_tail) {
    T tail_x_val = tail_x[global_thread_id];
    tail_y[global_thread_id] =
        tail_x_val > 0 ? tail_x_val
                       : tail_x_val * alpha[(global_thread_id / inner_size) % alpha_size];
  }
}

template<typename T, int pack_size, bool tail>
__global__ RETURN_VOID_IF_FLOAT PReluForwardMultiAlphaGpu(
    const int32_t elem_cnt, const int32_t alpha_size, const int32_t inner_size,
    const int64_t n_tail, const T* x, const T* alpha, T* y, const T* tail_x, T* tail_y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;

  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    LoadPack y_vec;
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      y_vec.elem[i] = x_vec.elem[i] > 0
                          ? x_vec.elem[i]
                          : x_vec.elem[i] * alpha[(linear_index / inner_size) % alpha_size];
    }
    *(reinterpret_cast<LoadType*>(y + linear_index)) = y_vec.storage;
  }

  if (tail && global_thread_id < n_tail) {
    T tail_x_val = tail_x[global_thread_id];
    tail_y[global_thread_id] =
        tail_x_val > 0 ? tail_x_val
                       : tail_x_val * alpha[(global_thread_id / inner_size) % alpha_size];
  }
}

template<typename T, int pack_size, bool tail>
__global__ RETURN_VOID_IF_HALF PReluBackwardMultiAlphaGpu(
    const int32_t elem_cnt, const int32_t alpha_size, const int32_t inner_size,
    const int32_t n_tail, const T* x, const T* alpha, const T* dy, T* dx, T* alpha_diff,
    const T* tail_x, const T* tail_dy, T* tail_dx) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t linear_index = global_thread_id; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x) {
    const T* x_load = x + linear_index;
    const T* dy_load = dy + linear_index;
    T* dx_load = dx + linear_index;
    dx_load[0] = x_load[0] > static_cast<T>(0.0)
                     ? dy_load[0]
                     : __hmul(dy_load[0], alpha[(linear_index / inner_size) % alpha_size]);
    alpha_diff[(linear_index / inner_size) % alpha_size] = __hadd(
        alpha_diff[(linear_index / inner_size) % alpha_size],
        x_load[0] > static_cast<T>(0.0) ? static_cast<T>(0.0) : __hmul(dy_load[0], x_load[0]));
  }

  if (tail && global_thread_id < n_tail) {
    const T* x_load = x + global_thread_id;
    const T* dy_load = dy + global_thread_id;
    T* dx_load = dx + global_thread_id;
    T* alpha_diff_load = alpha_diff + global_thread_id;
    dx_load[0] = x_load[0] > static_cast<T>(0.0)
                     ? dy_load[0]
                     : __hmul(dy_load[0], alpha[(global_thread_id / inner_size) % alpha_size]);
    alpha_diff_load[0] =
        __hadd(alpha_diff_load[0], x_load[0] > static_cast<T>(0.0) ? static_cast<T>(0.0)
                                                                   : __hmul(dy_load[0], x_load[0]));
  }
}

template<typename T, int pack_size, bool tail>
__global__ RETURN_VOID_IF_DOUBLE PReluBackwardMultiAlphaGpu(
    const int32_t elem_cnt, const int32_t alpha_size, const int32_t inner_size,
    const int32_t n_tail, const T* x, const T* alpha, const T* dy, T* dx, T* alpha_diff,
    const T* tail_x, const T* tail_dy, T* tail_dx) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;

  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    const LoadType* dy_load = reinterpret_cast<const LoadType*>(dy + linear_index);
    LoadPack dy_vec;
    dy_vec.storage = *dy_load;

    LoadPack dx_vec;
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      dx_vec.elem[i] = x_vec.elem[i] > 0
                           ? dy_vec.elem[i]
                           : dy_vec.elem[i] * alpha[(linear_index / inner_size) % alpha_size];
      alpha_diff[(linear_index / inner_size) % alpha_size] +=
          x_vec.elem[i] > 0 ? 0 : dy_vec.elem[i] * x_vec.elem[i];
    }

    *(reinterpret_cast<LoadType*>(dx + linear_index)) = dx_vec.storage;
  }

  if (tail && global_thread_id < n_tail) {
    T tail_x_val = tail_x[global_thread_id];
    T tail_dy_val = tail_dy[global_thread_id];
    tail_dx[global_thread_id] =
        tail_x_val > 0 ? tail_dy_val
                       : tail_dy_val * alpha[(global_thread_id / inner_size) % alpha_size];
    alpha_diff[(global_thread_id / inner_size) % alpha_size] +=
        tail_x_val > 0 ? 0 : tail_dy_val * tail_x[global_thread_id];
  }
}

template<typename T, int pack_size, bool tail>
__global__ RETURN_VOID_IF_FLOAT PReluBackwardMultiAlphaGpu(
    const int32_t elem_cnt, const int32_t alpha_size, const int32_t inner_size,
    const int32_t n_tail, const T* x, const T* alpha, const T* dy, T* dx, T* alpha_diff,
    const T* tail_x, const T* tail_dy, T* tail_dx) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;

  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    const LoadType* dy_load = reinterpret_cast<const LoadType*>(dy + linear_index);
    LoadPack dy_vec;
    dy_vec.storage = *dy_load;

    LoadPack dx_vec;
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      dx_vec.elem[i] = x_vec.elem[i] > 0
                           ? dy_vec.elem[i]
                           : dy_vec.elem[i] * alpha[(linear_index / inner_size) % alpha_size];
      alpha_diff[(linear_index / inner_size) % alpha_size] +=
          x_vec.elem[i] > 0 ? 0 : dy_vec.elem[i] * x_vec.elem[i];
    }

    *(reinterpret_cast<LoadType*>(dx + linear_index)) = dx_vec.storage;
  }

  if (tail && global_thread_id < n_tail) {
    T tail_x_val = tail_x[global_thread_id];
    T tail_dy_val = tail_dy[global_thread_id];
    tail_dx[global_thread_id] =
        tail_x_val > 0 ? tail_dy_val
                       : tail_dy_val * alpha[(global_thread_id / inner_size) % alpha_size];
    alpha_diff[(global_thread_id / inner_size) % alpha_size] +=
        tail_x_val > 0 ? 0 : tail_dy_val * tail_x[global_thread_id];
  }
}

template<int pack_size>
unsigned int ComputeGridSize(const int32_t block_size, const int64_t elem_cnt) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned int blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
  unsigned int grid_size = ((elem_cnt + block_size - 1) / block_size);
  grid_size = std::min((unsigned int)prop.multiProcessorCount * blocks_per_sm, grid_size);
  return grid_size;
}

template<typename T>
void DispatchForwardTail(ep::Stream* stream, const int32_t elem_cnt, const int32_t alpha_size,
                         const int32_t inner_size, const T* x, const T* alpha, T* y) {
  unsigned int grid_size = ComputeGridSize<4>(kBlockSize, elem_cnt);
  constexpr int pack_size = GetPreluPackSize<T>();
  const int64_t pack_num = elem_cnt / pack_size;
  const int64_t tail_offset = pack_num * pack_size;
  const int64_t n_tail = elem_cnt - tail_offset;
  const bool tail = n_tail > 0 ? true : false;
  uint64_t inc_offset = 0;

  if (tail) {
    // If tail, we need generate randnum one more time, so here we add another `1`.
    inc_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1) * kVecSize + 1;
    PReluForwardMultiAlphaGpu<T, pack_size, true>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, n_tail, x, alpha, y, (x + tail_offset),
            (y + tail_offset));
  } else {
    inc_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1) * kVecSize;
    PReluForwardMultiAlphaGpu<T, pack_size, false>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, n_tail, x, alpha, y, nullptr, nullptr);
  }
}

template<typename T>
void DispatchBackwardTail(ep::Stream* stream, const int32_t elem_cnt, const int32_t alpha_size,
                          const int32_t inner_size, const T* x, const T* alpha, const T* dy, T* dx,
                          T* alpha_diff) {
  unsigned int grid_size = ComputeGridSize<4>(kBlockSize, elem_cnt);
  constexpr int pack_size = GetPreluPackSize<T>();
  const int64_t pack_num = elem_cnt / pack_size;
  const int64_t tail_offset = pack_num * pack_size;
  const int64_t n_tail = elem_cnt - tail_offset;
  const bool tail = n_tail > 0 ? true : false;
  uint64_t inc_offset = 0;

  if (tail) {
    // If tail, we need generate randnum one more time, so here we add another `1`.
    inc_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1) * kVecSize + 1;
    PReluBackwardMultiAlphaGpu<T, pack_size, true>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, n_tail, x, alpha, dy, dx, alpha_diff,
            (x + tail_offset), (dy + tail_offset), (dx + tail_offset));
  } else {
    inc_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1) * kVecSize;
    PReluBackwardMultiAlphaGpu<T, pack_size, false>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, n_tail, x, alpha, dy, dx, alpha_diff, nullptr,
            nullptr, nullptr);
  }
}

template<typename T>
__global__ void PReluForwardSingleAlphaGpu(const int32_t elem_cnt, const T* x, const T* alpha,
                                           T* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { y[i] = x[i] > 0 ? x[i] : x[i] * alpha[0]; }
}

template<typename T>
__global__ void PReluBackwardSingleAlphaGpu(const int32_t elem_cnt, const T* x, const T* alpha,
                                            const T* dy, T* dx, T* alpha_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    dx[i] = x[i] > 0 ? dy[i] : dy[i] * alpha[0];
    alpha_diff[0] += x[i] > 0 ? 0 : dy[i] * x[i];
  }
}

template<>
__global__ void PReluForwardSingleAlphaGpu<half>(const int32_t elem_cnt, const half* x,
                                                 const half* alpha, half* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    y[i] = x[i] > static_cast<half>(0.0) ? x[i] : __hmul(x[i], alpha[0]);
  }
}

template<>
__global__ void PReluBackwardSingleAlphaGpu<half>(const int32_t elem_cnt, const half* x,
                                                  const half* alpha, const half* dy, half* dx,
                                                  half* alpha_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    dx[i] = x[i] > static_cast<half>(0.0) ? dy[i] : __hmul(dy[i], alpha[0]);
    alpha_diff[0] = __hadd(alpha_diff[0], x[i] > static_cast<half>(0.0) ? static_cast<half>(0.0)
                                                                        : __hmul(dy[i], x[i]));
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
    const int32_t alpha_size = alpha->shape().elem_cnt();
    if (alpha_size == 1) {
      PReluForwardSingleAlphaGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                      ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, x->dptr<T>(), alpha->dptr<T>(), y->mut_dptr<T>());
    } else {
      const int batch = x->shape().At(0);
      const int channels = x->shape().At(1);
      const int32_t inner_size = elem_cnt / batch / channels;
      DispatchForwardTail<T>(
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

    const int32_t elem_cnt = x->shape().elem_cnt();
    const int32_t alpha_size = alpha->shape().elem_cnt();

    if (alpha_size == 1) {
      PReluBackwardSingleAlphaGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                       ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, x->dptr<T>(), alpha->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>(),
          alpha_diff->mut_dptr<T>());
    } else {
      const int batch = x->shape().At(0);
      const int channels = x->shape().At(1);
      const int32_t inner_size = elem_cnt / batch / channels;

      DispatchBackwardTail<T>(
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
