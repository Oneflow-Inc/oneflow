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
#include <cub/cub.cuh>
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/user_op_hob.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<typename T>
static __forceinline__ __device__ T sign(T val) {
  return (0 < val) - (val < 0);
}

template<typename T>
static __forceinline__ __device__ T device_sqrt(T val);

template<>
__forceinline__ __device__ float device_sqrt(float val) {
  return ::sqrtf(val);
}

template<>
__forceinline__ __device__ double device_sqrt(double val) {
  return ::sqrt(val);
}

// Zero norm
template<typename T>
struct ZeroDist {
  static __forceinline__ __device__ void inc(T& agg, const T diff, const double /*p*/) {
    agg += diff != 0.0;
  }
  static __forceinline__ __device__ T finish(const T agg, const double /*p*/) { return agg; }
  static __forceinline__ __device__ void agg(T& update, const T other) { update += other; }
};

// One norm
template<typename T>
struct OneDist {
  static __forceinline__ __device__ void inc(T& agg, const T diff, const double /*p*/) {
    agg += diff;
  }
  static __forceinline__ __device__ T finish(const T agg, const double /*p*/) { return agg; }
  static __forceinline__ __device__ void agg(T& update, const T other) { update += other; }
  static __forceinline__ __device__ T backward(const T diff, const T grad, const T /*dist*/,
                                               const double /*p*/) {
    return grad * sign(diff);
  }
};

// Two norm
template<typename T>
struct TwoDist {
  static __forceinline__ __device__ void inc(T& agg, const T diff, const double /*p*/) {
    agg += diff * diff;
  }
  static __forceinline__ __device__ T finish(const T agg, const double /*p*/) {
    return device_sqrt<T>(agg);
  }
  static __forceinline__ __device__ void agg(T& update, const T other) { update += other; }
  static __forceinline__ __device__ T backward(const T diff, const T grad, const T dist,
                                               const double /*p*/) {
    return dist == 0.0 ? 0 : grad * diff / dist;
  }
};

// General p norm
template<typename T>
struct PDist {
  static __forceinline__ __device__ void inc(T& agg, const T diff, const double p) {
    agg += std::pow(diff, p);
  }
  static __forceinline__ __device__ T finish(const T agg, const double p) {
    return std::pow(agg, static_cast<double>(1) / p);
  }
  static __forceinline__ __device__ void agg(T& update, const T other) { update += other; }
  static __forceinline__ __device__ T backward(const T diff, const T grad, const T dist,
                                               const double p) {
    return dist == 0.0 ? 0 : diff * std::pow(std::abs(diff), p - 2) * grad / std::pow(dist, p - 1);
  }
};

// Inf norm
template<typename T>
struct InfiDist {
  static __forceinline__ __device__ void inc(T& agg, const T diff, const double /*p*/) {
    if (diff > agg) { agg = diff; }
  }
  static __forceinline__ __device__ T finish(const T agg, const double /*p*/) { return agg; }
  static __forceinline__ __device__ void agg(T& update, const T other) {
    if (other > update) { update = other; }
  }
  static __forceinline__ __device__ T backward(const T diff, const T grad, const T dist,
                                               const double /*p*/) {
    return grad * sign(diff) * (std::abs(diff) == dist);
  }
};

template<typename T, typename Dist>
struct DistReduce {
  __forceinline__ __device__ T operator()(T a, T b) const {
    Dist::agg(a, b);
    return a;
  }
};

template<typename T>
__global__ static void reduce_backward_buffer(T* buffer, T* grad, int64_t reduce_size) {
  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  int32_t row_idx = blockIdx.x;
  int32_t col_idx = threadIdx.x;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T agg = 0;
  for(int32_t col = col_idx; col < reduce_size; col += blockDim.x) {
    int idx = row_idx * reduce_size + col_idx;
    agg += buffer[idx];
  }
  T result = BlockReduce(temp_storage).Sum(agg);
  if (threadIdx.x == 0) { grad[blockIdx.x] = result; }
}


template<typename T, typename Dist>
__global__ static void CUDACDistForward(const T* x1, const T* x2, T* out, int64_t r1, int64_t r2,
                                        int64_t c, int64_t r_size, int64_t r1_size, int64_t r2_size,
                                        double p) {
  const int64_t batch_idx = blockIdx.x / r_size;
  const int64_t vec_out_idx = blockIdx.x - batch_idx * r_size;
  const int64_t vec1_idx = vec_out_idx / r2;
  const int64_t vec2_idx = vec_out_idx - vec1_idx * r2;
  const int64_t stride = blockDim.x;

  const T* vec1_begin = x1 + batch_idx * r1_size + vec1_idx * c + threadIdx.x;
  const T* vec1_end = x1 + batch_idx * r1_size + vec1_idx * c + c;
  const T* vec2_begin = x2 + batch_idx * r2_size + vec2_idx * c + threadIdx.x;

  T agg = 0;
  for (; vec1_begin < vec1_end; vec1_begin += stride, vec2_begin += stride) {
    Dist::inc(agg, std::abs(*vec1_begin - *vec2_begin), p);
  }

  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T result = BlockReduce(temp_storage).Reduce(agg, DistReduce<T, Dist>());
  if (threadIdx.x == 0) { out[blockIdx.x] = Dist::finish(result, p); }
}

template<typename T, typename Dist>
__global__ static void CUDACDistBackward(const T* x1, const T* x2, const T* dist, const T* dist_grad,
                                         T* grad1, T* grad2, int64_t r1, int64_t r2, int64_t c,
                                         int64_t r_size, int64_t r1_size, int64_t r2_size,
                                         double p, T* buffer1, T* buffer2) {
  const int64_t batch_idx = blockIdx.x / r_size;
  const int64_t vec_out_idx = blockIdx.x - batch_idx * r_size;
  const int64_t vec1_idx = vec_out_idx / r2;
  const int64_t vec2_idx = vec_out_idx - vec1_idx * r2;
  const int64_t stride = blockDim.x;

  const T* vec1_begin = x1 + batch_idx * r1_size + vec1_idx * c + threadIdx.x;
  const T* vec1_end = x1 + batch_idx * r1_size + vec1_idx * c + c;
  const T* vec2_begin = x2 + batch_idx * r2_size + vec2_idx * c + threadIdx.x;

  T* grad1_begin = vec1_begin - x1 + grad1;
  T* grad2_begin = vec2_begin - x2 + grad2;
  T diff = *vec1_begin - *vec2_begin;

  T* buffer1_idx = buffer1 + batch_idx * r_size * c + vec1_idx * r2 * c + threadIdx.x * r2 + vec2_idx;
  T* buffer2_idx = buffer2 + batch_idx * r_size * c + vec2_idx * r1 * c + threadIdx.x * r1 + vec1_idx;
  *buffer1_idx = Dist::backward(diff, *(dist_grad + blockIdx.x), *(dist + blockIdx.x), p);
  *buffer2_idx = Dist::backward(-diff, *(dist_grad + blockIdx.x), *(dist + blockIdx.x), p);
}

template<typename T>
class CUDACDistKernel final : public user_op::OpKernel {
 public:
  CUDACDistKernel() = default;
  ~CUDACDistKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x1 = ctx->Tensor4ArgNameAndIndex("x1", 0);
    const user_op::Tensor* x2 = ctx->Tensor4ArgNameAndIndex("x2", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    double p = ctx->Attr<double>("p");
    int64_t ndim = x1->shape_view().NumAxes();
    int64_t r1 = x1->shape_view().At(ndim - 2);
    int64_t r2 = x2->shape_view().At(ndim - 2);
    int64_t c = x1->shape_view().At(ndim - 1);

    const int64_t r1_size = r1 * c;
    const int64_t r2_size = r2 * c;
    const int64_t r_size = r1 * r2;

    const T* x1_ptr = x1->dptr<T>();
    const T* x2_ptr = x2->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    if (p == 0) {
      CUDACDistForward<T, ZeroDist<T>><<<out->shape_view().elem_cnt(), kCudaThreadsNumPerBlock, 0,
                                         ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          x1_ptr, x2_ptr, out_ptr, r1, r2, c, r_size, r1_size, r2_size, p);
    } else if (p == 1) {
      CUDACDistForward<T, OneDist<T>><<<out->shape_view().elem_cnt(), kCudaThreadsNumPerBlock, 0,
                                        ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          x1_ptr, x2_ptr, out_ptr, r1, r2, c, r_size, r1_size, r2_size, p);
    } else if (p == 2) {
      CUDACDistForward<T, TwoDist<T>><<<out->shape_view().elem_cnt(), kCudaThreadsNumPerBlock, 0,
                                        ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          x1_ptr, x2_ptr, out_ptr, r1, r2, c, r_size, r1_size, r2_size, p);
    } else if (std::isinf(p)) {
      CUDACDistForward<T, InfiDist<T>><<<out->shape_view().elem_cnt(), kCudaThreadsNumPerBlock, 0,
                                         ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          x1_ptr, x2_ptr, out_ptr, r1, r2, c, r_size, r1_size, r2_size, p);
    } else {
      CUDACDistForward<T, PDist<T>><<<out->shape_view().elem_cnt(), kCudaThreadsNumPerBlock, 0,
                                      ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          x1_ptr, x2_ptr, out_ptr, r1, r2, c, r_size, r1_size, r2_size, p);
    }

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class CUDACDistGradKernel final : public user_op::OpKernel {
 public:
  CUDACDistGradKernel() = default;
  ~CUDACDistGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x1 = ctx->Tensor4ArgNameAndIndex("x1", 0);
    const user_op::Tensor* x2 = ctx->Tensor4ArgNameAndIndex("x2", 0);
    const user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx1 = ctx->Tensor4ArgNameAndIndex("dx1", 0);
    user_op::Tensor* dx2 = ctx->Tensor4ArgNameAndIndex("dx2", 0);
    double p = ctx->Attr<double>("p");
    int64_t ndim = x1->shape_view().NumAxes();
    int64_t r1 = x1->shape_view().At(ndim - 2);
    int64_t r2 = x2->shape_view().At(ndim - 2);
    int64_t c = x1->shape_view().At(ndim - 1);

    const T* x1_ptr = x1->dptr<T>();
    const T* x2_ptr = x2->dptr<T>();
    const T* dist_ptr = out->dptr<T>();
    const T* grad_ptr = dy->dptr<T>();

    const int64_t r1_size = r1 * c;
    const int64_t r2_size = r2 * c;
    const int64_t r_size = r1 * r2;

    T* dx1_ptr = dx1->mut_dptr<T>();
    T* dx2_ptr = dx2->mut_dptr<T>();

    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), dx1_ptr, 0, dx1->shape_view().elem_cnt() * sizeof(T));
    memset_primitive->Launch(ctx->stream(), dx2_ptr, 0, dx2->shape_view().elem_cnt() * sizeof(T));


    T* buffer1 = nullptr;
    T* buffer2 = nullptr;
    OF_CUDA_CHECK(cudaMalloc(&buffer1, out->shape_view().elem_cnt() * c * sizeof(T)));
    OF_CUDA_CHECK(cudaMalloc(&buffer2, out->shape_view().elem_cnt() * c * sizeof(T)));

    if (p == 0) {
      // grad is always zero
    } else if (p == 1) {
      CUDACDistBackward<T, OneDist<T>><<<out->shape_view().elem_cnt(), c, 0,
                                         ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          x1_ptr, x2_ptr, dist_ptr, grad_ptr, dx1_ptr, dx2_ptr, r1, r2, c, r_size, r1_size, r2_size,
          p, buffer1, buffer2);
    } else if (p == 2) {
      CUDACDistBackward<T, TwoDist<T>><<<out->shape_view().elem_cnt(), c, 0,
                                         ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          x1_ptr, x2_ptr, dist_ptr, grad_ptr, dx1_ptr, dx2_ptr, r1, r2, c, r_size, r1_size, r2_size,
          p, buffer1, buffer2);
    } else if (std::isinf(p)) {
      CUDACDistBackward<T, InfiDist<T>><<<out->shape_view().elem_cnt(), c, 0,
                                          ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          x1_ptr, x2_ptr, dist_ptr, grad_ptr, dx1_ptr, dx2_ptr, r1, r2, c, r_size, r1_size, r2_size,
          p, buffer1, buffer2);
    } else {
      CUDACDistBackward<T, PDist<T>><<<out->shape_view().elem_cnt(), c, 0,
                                       ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          x1_ptr, x2_ptr, dist_ptr, grad_ptr, dx1_ptr, dx2_ptr, r1, r2, c, r_size, r1_size, r2_size,
          p, buffer1, buffer2);
    }
    reduce_backward_buffer<T><<<dx1->shape_view().elem_cnt(), kCudaThreadsNumPerBlock, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(buffer1, dx1_ptr, r2);
    reduce_backward_buffer<T><<<dx2->shape_view().elem_cnt(), kCudaThreadsNumPerBlock, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(buffer2, dx2_ptr, r1);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_CDIST_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("cdist").SetCreateFn<CUDACDistKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                  \
      && (user_op::HobDataType("x1", 0) == GetDataType<dtype>::value)                  \
      && (user_op::HobDataType("x2", 0) == GetDataType<dtype>::value)                  \
      && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CDIST_KERNEL(float)
REGISTER_CUDA_CDIST_KERNEL(double)
#undef REGISTER_CUDA_CDIST_KERNEL

#define REGISTER_CUDA_CDIST_GRAD_KERNEL(dtype)                                         \
  REGISTER_USER_KERNEL("cdist_grad")                                                   \
      .SetCreateFn<CUDACDistGradKernel<dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("x1", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("x2", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CDIST_GRAD_KERNEL(float)
REGISTER_CUDA_CDIST_GRAD_KERNEL(double)
#undef REGISTER_CUDA_CDIST_KERNEL

}  // namespace oneflow
