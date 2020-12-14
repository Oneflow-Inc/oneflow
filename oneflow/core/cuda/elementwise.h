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
#ifndef ONEFLOW_CORE_CUDA_ELEMENTWISE_H_
#define ONEFLOW_CORE_CUDA_ELEMENTWISE_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>
#include <type_traits>

namespace oneflow {

namespace cuda {

namespace elementwise {

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 64;

inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize, sm_count * kNumWaves));
  return cudaSuccess;
}

template<typename T, int pack_size>
struct SOA {
  T array[pack_size];
};

template<typename T, int pack_size>
struct GetPackType {
  using type = SOA<T, pack_size>;
};

template<typename T>
struct GetPackType<T, 1> {
  using type = T;
};

template<>
struct GetPackType<half, 2> {
  using type = half2;
};

template<>
struct GetPackType<half, 4> {
  using type = short4;
};

template<>
struct GetPackType<half, 8> {
  using type = ulonglong2;
};

template<>
struct GetPackType<float, 2> {
  using type = float2;
};

template<>
struct GetPackType<float, 4> {
  using type = float4;
};

template<>
struct GetPackType<double, 2> {
  using type = double2;
};

template<>
struct GetPackType<int8_t, 2> {
  using type = char2;
};

template<>
struct GetPackType<int8_t, 4> {
  using type = char4;
};

template<>
struct GetPackType<int8_t, 8> {
  using type = long;
};

template<>
struct GetPackType<int32_t, 2> {
  using type = int2;
};

template<>
struct GetPackType<int32_t, 4> {
  using type = int4;
};

template<>
struct GetPackType<int64_t, 2> {
  using type = longlong2;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, int pack_size,
         typename std::enable_if<(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size)>::type* =
             nullptr>
union Pack {
  __device__ Pack() {
    // do nothing
  }
  PackType<T, pack_size> storage;
  T elem[pack_size];
};

constexpr int kMaxPackBytes = 128 / 8;
constexpr int kMaxPackSize = 8;

constexpr int Min(int a, int b) { return a < b ? a : b; }

template<typename T>
constexpr int PackSize() {
  return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

template<typename T, typename... Args,
         typename std::enable_if<(sizeof...(Args) >= 1)>::type* = nullptr>
constexpr int PackSize() {
  return Min(PackSize<T>(), PackSize<Args...>());
}

template<typename F, typename R, typename X, int pack_size, bool tail>
__global__ void __launch_bounds__(kBlockSize)
    ApplyUnary(F functor, int64_t n_pack, const PackType<X, pack_size>* pack_x,
               PackType<R, pack_size>* pack_r, int64_t n_tail, const X* tail_x, R* tail_r) {
  Pack<X, pack_size> p_x;
  Pack<R, pack_size> p_r;
  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    p_x.storage = pack_x[i];
#pragma unroll
    for (int j = 0; j < pack_size; ++j) { p_r.elem[j] = functor(p_x.elem[j]); }
    pack_r[i] = p_r.storage;
  }
  if (tail && global_tid < n_tail) { tail_r[global_tid] = functor(tail_x[global_tid]); }
}

template<typename F, typename R, typename X, typename Y, int pack_size, bool tail>
__global__ void __launch_bounds__(kBlockSize)
    ApplyBinary(F functor, int64_t n_pack, const PackType<X, pack_size>* pack_x,
                const PackType<Y, pack_size>* pack_y, PackType<R, pack_size>* pack_r,
                int64_t n_tail, const X* tail_x, const Y* tail_y, R* tail_r) {
  Pack<X, pack_size> p_x;
  Pack<Y, pack_size> p_y;
  Pack<R, pack_size> p_r;
  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    p_x.storage = pack_x[i];
    p_y.storage = pack_y[i];
#pragma unroll
    for (int j = 0; j < pack_size; ++j) { p_r.elem[j] = functor(p_x.elem[j], p_y.elem[j]); }
    pack_r[i] = p_r.storage;
  }
  if (tail && global_tid < n_tail) {
    tail_r[global_tid] = functor(tail_x[global_tid], tail_y[global_tid]);
  }
}

template<typename F, typename R, typename X, typename Y, typename Z, int pack_size, bool tail>
__global__ void __launch_bounds__(kBlockSize)
    ApplyTernary(F functor, int64_t n_pack, const PackType<X, pack_size>* pack_x,
                 const PackType<Y, pack_size>* pack_y, const PackType<Z, pack_size>* pack_z,
                 PackType<R, pack_size>* pack_r, int64_t n_tail, const X* tail_x, const Y* tail_y,
                 const Z* tail_z, R* tail_r) {
  Pack<X, pack_size> p_x;
  Pack<Y, pack_size> p_y;
  Pack<Z, pack_size> p_z;
  Pack<R, pack_size> p_r;
  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    p_x.storage = pack_x[i];
    p_y.storage = pack_y[i];
    p_z.storage = pack_z[i];
#pragma unroll
    for (int j = 0; j < pack_size; ++j) {
      p_r.elem[j] = functor(p_x.elem[j], p_y.elem[j], p_z.elem[j]);
    }
    pack_r[i] = p_r.storage;
  }
  if (tail && global_tid < n_tail) {
    tail_r[global_tid] = functor(tail_x[global_tid], tail_y[global_tid], tail_z[global_tid]);
  }
}

template<typename F, typename R, typename X>
struct Unary {
  static cudaError_t Launch(F functor, int64_t n, const X* x, R* r, cudaStream_t stream) {
    constexpr int pack_size = PackSize<R, X>();
    const int64_t n_pack = n / pack_size;
    const int64_t tail_offset = n_pack * pack_size;
    const int64_t n_tail = n - tail_offset;
    int num_blocks;
    {
      cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
      if (err != cudaSuccess) { return err; }
    }
    if (n_tail > 0) {
      ApplyUnary<F, R, X, pack_size, true><<<num_blocks, kBlockSize, 0, stream>>>(
          functor, n_pack, reinterpret_cast<const PackType<X, pack_size>*>(x),
          reinterpret_cast<PackType<R, pack_size>*>(r), n_tail, x + tail_offset, r + tail_offset);
    } else {
      ApplyUnary<F, R, X, pack_size, false><<<num_blocks, kBlockSize, 0, stream>>>(
          functor, n_pack, reinterpret_cast<const PackType<X, pack_size>*>(x),
          reinterpret_cast<PackType<R, pack_size>*>(r), n_tail, nullptr, nullptr);
    }
    return cudaSuccess;
  }
};

template<typename F, typename R, typename X, typename Y>
struct Binary {
  static cudaError_t Launch(F functor, int64_t n, const X* x, const Y* y, R* r,
                            cudaStream_t stream) {
    constexpr int pack_size = PackSize<R, X, Y>();
    const int64_t n_pack = n / pack_size;
    const int64_t tail_offset = n_pack * pack_size;
    const int64_t n_tail = n - tail_offset;
    int num_blocks;
    {
      cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
      if (err != cudaSuccess) { return err; }
    }
    if (n_tail > 0) {
      ApplyBinary<F, R, X, Y, pack_size, true><<<num_blocks, kBlockSize, 0, stream>>>(
          functor, n_pack, reinterpret_cast<const PackType<X, pack_size>*>(x),
          reinterpret_cast<const PackType<Y, pack_size>*>(y),
          reinterpret_cast<PackType<R, pack_size>*>(r), n_tail, x + tail_offset, y + tail_offset,
          r + tail_offset);
    } else {
      ApplyBinary<F, R, X, Y, pack_size, false><<<num_blocks, kBlockSize, 0, stream>>>(
          functor, n_pack, reinterpret_cast<const PackType<X, pack_size>*>(x),
          reinterpret_cast<const PackType<Y, pack_size>*>(y),
          reinterpret_cast<PackType<R, pack_size>*>(r), n_tail, nullptr, nullptr, nullptr);
    }
    return cudaSuccess;
  }
};

template<typename F, typename R, typename X, typename Y, typename Z>
struct Ternary {
  static cudaError_t Launch(F functor, int64_t n, const X* x, const Y* y, const Z* z, R* r,
                            cudaStream_t stream) {
    constexpr int pack_size = PackSize<R, X, Y>();
    const int64_t n_pack = n / pack_size;
    const int64_t tail_offset = n_pack * pack_size;
    const int64_t n_tail = n - tail_offset;
    int num_blocks;
    {
      cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
      if (err != cudaSuccess) { return err; }
    }
    if (n_tail > 0) {
      ApplyTernary<F, R, X, Y, Z, pack_size, true><<<num_blocks, kBlockSize, 0, stream>>>(
          functor, n_pack, reinterpret_cast<const PackType<X, pack_size>*>(x),
          reinterpret_cast<const PackType<Y, pack_size>*>(y),
          reinterpret_cast<const PackType<Z, pack_size>*>(z),
          reinterpret_cast<PackType<R, pack_size>*>(r), n_tail, x + tail_offset, y + tail_offset,
          z + tail_offset, r + tail_offset);
    } else {
      ApplyTernary<F, R, X, Y, Z, pack_size, false><<<num_blocks, kBlockSize, 0, stream>>>(
          functor, n_pack, reinterpret_cast<const PackType<X, pack_size>*>(x),
          reinterpret_cast<const PackType<Y, pack_size>*>(y),
          reinterpret_cast<const PackType<Z, pack_size>*>(z),
          reinterpret_cast<PackType<R, pack_size>*>(r), n_tail, nullptr, nullptr, nullptr, nullptr);
    }
    return cudaSuccess;
  }
};

}  // namespace elementwise

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_ELEMENTWISE_H_
