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
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                   sm_count * tpm / kBlockSize * kNumWaves));
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

template<typename T, int pack_size>
union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
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

template<typename T, typename U, typename... Args>
constexpr int PackSize() {
  return Min(PackSize<T>(), PackSize<U, Args...>());
}

template<typename FactoryT, typename R, typename X, int pack_size, bool tail>
__global__ void __launch_bounds__(kBlockSize)
    ApplyUnary(FactoryT factory, int64_t n_pack, const PackType<X, pack_size>* pack_x,
               PackType<R, pack_size>* pack_r, int64_t n_tail, const X* tail_x, R* tail_r) {
  auto functor = factory.Create();
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

template<int pack_size, bool tail, typename FactoryT, typename R, typename X>
struct UnaryLauncher {
  static void Launch(int num_blocks, cudaStream_t stream, FactoryT factory, int64_t n_pack,
                     const PackType<X, pack_size>* pack_x, PackType<R, pack_size>* pack_r,
                     int64_t n_tail, const X* tail_x, R* tail_r) {
    ApplyUnary<FactoryT, R, X, pack_size, tail><<<num_blocks, kBlockSize, 0, stream>>>(
        factory, n_pack, pack_x, pack_r, n_tail, tail_x, tail_r);
  }
};

template<typename FactoryT, typename R, typename X, typename Y, int pack_size, bool tail>
__global__ void __launch_bounds__(kBlockSize)
    ApplyBinary(FactoryT factory, int64_t n_pack, const PackType<X, pack_size>* pack_x,
                const PackType<Y, pack_size>* pack_y, PackType<R, pack_size>* pack_r,
                int64_t n_tail, const X* tail_x, const Y* tail_y, R* tail_r) {
  auto functor = factory.Create();
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

template<int pack_size, bool tail, typename FactoryT, typename R, typename X, typename Y>
struct BinaryLauncher {
  static void Launch(int num_blocks, cudaStream_t stream, FactoryT factory, int64_t n_pack,
                     const PackType<X, pack_size>* pack_x, const PackType<Y, pack_size>* pack_y,
                     PackType<R, pack_size>* pack_r, int64_t n_tail, const X* tail_x,
                     const Y* tail_y, R* tail_r) {
    ApplyBinary<FactoryT, R, X, Y, pack_size, tail><<<num_blocks, kBlockSize, 0, stream>>>(
        factory, n_pack, pack_x, pack_y, pack_r, n_tail, tail_x, tail_y, tail_r);
  }
};

template<typename FactoryT, typename R, typename X, typename Y, typename Z, int pack_size,
         bool tail>
__global__ void __launch_bounds__(kBlockSize)
    ApplyTernary(FactoryT factory, int64_t n_pack, const PackType<X, pack_size>* pack_x,
                 const PackType<Y, pack_size>* pack_y, const PackType<Z, pack_size>* pack_z,
                 PackType<R, pack_size>* pack_r, int64_t n_tail, const X* tail_x, const Y* tail_y,
                 const Z* tail_z, R* tail_r) {
  auto functor = factory.Create();
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

template<int pack_size, bool tail, typename FactoryT, typename R, typename X, typename Y,
         typename Z>
struct TernaryLauncher {
  static void Launch(int num_blocks, cudaStream_t stream, FactoryT factory, int64_t n_pack,
                     const PackType<X, pack_size>* pack_x, const PackType<Y, pack_size>* pack_y,
                     const PackType<Z, pack_size>* pack_z, PackType<R, pack_size>* pack_r,
                     int64_t n_tail, const X* tail_x, const Y* tail_y, const Z* tail_z, R* tail_r) {
    ApplyTernary<FactoryT, R, X, Y, Z, pack_size, tail><<<num_blocks, kBlockSize, 0, stream>>>(
        factory, n_pack, pack_x, pack_y, pack_z, pack_r, n_tail, tail_x, tail_y, tail_z, tail_r);
  }
};

template<typename T, int pack_size>
inline const PackType<T, pack_size>* ToPackType(const T* ptr) {
  return reinterpret_cast<const PackType<T, pack_size>*>(ptr);
}

template<typename T, int pack_size>
inline PackType<T, pack_size>* ToPackType(T* ptr) {
  return reinterpret_cast<PackType<T, pack_size>*>(ptr);
}

template<typename FunctorT>
struct SimpleFactory {
  explicit SimpleFactory(FunctorT functor) : functor_(functor) {}
  __device__ FunctorT Create() const { return functor_; }

 private:
  FunctorT functor_;
};

template<template<int, bool, typename, typename, typename...> typename LauncherT, typename FactoryT,
         typename R, typename... IN>
struct GenericLauncher {
  constexpr static int pack_size = PackSize<R, IN...>();
  static cudaError_t Launch(FactoryT factory, int64_t n, const IN*... in, R* r,
                            cudaStream_t stream) {
    constexpr int pack_size = PackSize<R, IN...>();
    const int64_t n_pack = n / pack_size;
    const int64_t tail_offset = n_pack * pack_size;
    const int64_t n_tail = n - tail_offset;
    int num_blocks;
    {
      cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
      if (err != cudaSuccess) { return err; }
    }
    auto func = n_tail > 0 ? LauncherT<pack_size, true, FactoryT, R, IN...>::Launch
                           : LauncherT<pack_size, false, FactoryT, R, IN...>::Launch;
    func(num_blocks, stream, factory, n_pack, ToPackType<IN, pack_size>(in)...,
         ToPackType<R, pack_size>(r), n_tail, (in + tail_offset)..., r + tail_offset);
    return cudaPeekAtLastError();
  }
};

template<typename R, typename X>
struct Unary {
  template<typename FunctorT>
  static cudaError_t Launch(FunctorT functor, int64_t n, const X* x, R* r, cudaStream_t stream) {
    return LaunchWithFactory(SimpleFactory<FunctorT>(functor), n, x, r, stream);
  }
  template<typename FactoryT>
  static cudaError_t LaunchWithFactory(FactoryT factory, int64_t n, const X* x, R* r,
                                       cudaStream_t stream) {
    return GenericLauncher<UnaryLauncher, FactoryT, R, X>::Launch(factory, n, x, r, stream);
  }
};

template<typename R, typename X, typename Y>
struct Binary {
  template<typename FunctorT>
  static cudaError_t Launch(FunctorT functor, int64_t n, const X* x, const Y* y, R* r,
                            cudaStream_t stream) {
    return LaunchWithFactory(SimpleFactory<FunctorT>(functor), n, x, y, r, stream);
  }
  template<typename FactoryT>
  static cudaError_t LaunchWithFactory(FactoryT factory, int64_t n, const X* x, const Y* y, R* r,
                                       cudaStream_t stream) {
    return GenericLauncher<BinaryLauncher, FactoryT, R, X, Y>::Launch(factory, n, x, y, r, stream);
  }
};

template<typename R, typename X, typename Y, typename Z>
struct Ternary {
  template<typename FunctorT>
  static cudaError_t Launch(FunctorT functor, int64_t n, const X* x, const Y* y, const Z* z, R* r,
                            cudaStream_t stream) {
    return LaunchWithFactory(SimpleFactory<FunctorT>(functor), n, x, y, z, r, stream);
  }
  template<typename FactoryT>
  static cudaError_t LaunchWithFactory(FactoryT factory, int64_t n, const X* x, const Y* y,
                                       const Z* z, R* r, cudaStream_t stream) {
    return GenericLauncher<TernaryLauncher, FactoryT, R, X, Y, Z>::Launch(factory, n, x, y, z, r,
                                                                          stream);
  }
};

}  // namespace elementwise

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_ELEMENTWISE_H_
