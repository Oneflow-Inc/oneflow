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
#include <cstdint>
#include <algorithm>
#include <type_traits>

namespace oneflow {

namespace cuda {

namespace elementwise {

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;

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
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
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

template<typename T, int pack_size>
__device__ inline Pack<T, pack_size> FetchPack(const PackType<T, pack_size>* ptr) {
  Pack<T, pack_size> pack;
  pack.storage = *ptr;
  return pack;
}

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

template<int pack_size, typename FunctorT, typename R, typename... IN>
__device__ PackType<R, pack_size> ApplyPack(const FunctorT& functor, const IN... in[pack_size]) {
  Pack<R, pack_size> ret;
#pragma unroll
  for (int j = 0; j < pack_size; ++j) { ret.elem[j] = functor((in[j])...); }
  return ret.storage;
}

template<int pack_size, bool tail, typename FactoryT, typename R, typename... IN>
__global__ void __launch_bounds__(kBlockSize)
    ApplyGeneric(FactoryT factory, int64_t n_pack, PackType<R, pack_size>* pack_r,
                 const PackType<IN, pack_size>*... pack_in, int64_t n_tail, R* tail_r,
                 const IN*... tail_in) {
  auto functor = factory();
  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    pack_r[i] = ApplyPack<pack_size, decltype(functor), R, IN...>(
        functor, (FetchPack<IN, pack_size>(pack_in + i).elem)...);
  }
  if (tail && global_tid < n_tail) { tail_r[global_tid] = functor((tail_in[global_tid])...); }
}

template<typename FunctorT>
struct SimpleFactory {
  explicit SimpleFactory(FunctorT functor) : tpl(functor) {}
  __device__ FunctorT operator()() const { return tpl; }

 private:
  FunctorT tpl;
};

template<typename FactoryT, typename R, typename... IN>
struct GenericLauncher {
  static cudaError_t Launch(FactoryT factory, int64_t n, R* r, const IN*... in,
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
    auto func = n_tail > 0 ? ApplyGeneric<pack_size, true, FactoryT, R, IN...>
                           : ApplyGeneric<pack_size, false, FactoryT, R, IN...>;
    func<<<num_blocks, kBlockSize, 0, stream>>>(
        factory, n_pack, reinterpret_cast<PackType<R, pack_size>*>(r),
        (reinterpret_cast<const PackType<IN, pack_size>*>(in))..., n_tail, r + tail_offset,
        (in + tail_offset)...);
    return cudaPeekAtLastError();
  }
};

template<typename FactoryT, typename R, typename A>
inline cudaError_t UnaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a,
                                    cudaStream_t stream) {
  return GenericLauncher<FactoryT, R, A>::Launch(factory, n, r, a, stream);
}

template<typename FunctorT, typename R, typename A>
inline cudaError_t Unary(FunctorT functor, int64_t n, R* r, const A* a, cudaStream_t stream) {
  return UnaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, stream);
}

template<typename FactoryT, typename R, typename A, typename B>
inline cudaError_t BinaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a, const B* b,
                                     cudaStream_t stream) {
  return GenericLauncher<FactoryT, R, A, B>::Launch(factory, n, r, a, b, stream);
}

template<typename FunctorT, typename R, typename A, typename B>
inline cudaError_t Binary(FunctorT functor, int64_t n, R* r, const A* a, const B* b,
                          cudaStream_t stream) {
  return BinaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, b, stream);
}

template<typename FactoryT, typename R, typename A, typename B, typename C>
inline cudaError_t TernaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a, const B* b,
                                      const C* c, cudaStream_t stream) {
  return GenericLauncher<FactoryT, R, A, B, C>::Launch(factory, n, r, a, b, c, stream);
}

template<typename FunctorT, typename R, typename A, typename B, typename C>
inline cudaError_t Ternary(FunctorT functor, int64_t n, R* r, const A* a, const B* b, const C* c,
                           cudaStream_t stream) {
  return TernaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, b, c, stream);
}

}  // namespace elementwise

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_ELEMENTWISE_H_
