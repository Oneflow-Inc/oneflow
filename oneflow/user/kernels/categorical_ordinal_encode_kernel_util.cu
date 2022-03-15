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
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include "oneflow/user/kernels/categorical_ordinal_encode_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

using CuInt64T = unsigned long long int;

__device__ __inline__ int32_t AtomicCAS(int32_t* address, int32_t compare, int32_t val) {
  return atomicCAS(address, compare, val);
}

__device__ __inline__ int64_t AtomicCAS(int64_t* address, int64_t compare, int64_t val) {
  static_assert(sizeof(int64_t) == sizeof(CuInt64T), "size error");
  return static_cast<int64_t>(atomicCAS(reinterpret_cast<CuInt64T*>(address),
                                        static_cast<CuInt64T>(compare),
                                        static_cast<CuInt64T>(val)));
}

__device__ __inline__ int32_t AtomicAdd(int32_t* address, int32_t val) {
  return atomicAdd(address, val);
}

__device__ __inline__ int64_t AtomicAdd(int64_t* address, int64_t val) {
  static_assert(sizeof(int64_t) == sizeof(CuInt64T), "size error");
  return static_cast<int64_t>(
      atomicAdd(reinterpret_cast<CuInt64T*>(address), static_cast<CuInt64T>(val)));
}

template<typename K, typename V>
__device__ bool TryGetOrInsert(K* key, volatile V* value, V* size, const K hash, V* out) {
  K old_key = AtomicCAS(key, static_cast<K>(0), hash);
  if (old_key == 0) {
    V v = AtomicAdd(size, 1) + 1;
    *value = v;
    *out = v;
    return true;
  } else if (old_key == hash) {
    while (true) {
      V v = *value;
      if (v != 0) {
        *out = v;
        break;
      }
    }
    return true;
  } else {
    return false;
  }
}

template<typename T>
__device__ bool GetOrInsertOne(const size_t capacity, T* table, T* size, const T hash, T* out) {
  if (hash == 0) {
    *out = 0;
    return true;
  }
  const size_t start_idx = static_cast<size_t>(hash) % capacity;
  // fast path
  {
    T* key = table + start_idx * 2;
    T* value = key + 1;
    if (*key == hash && *value != 0) {
      *out = *value;
      return true;
    }
  }
  for (size_t count = 0; count < capacity; ++count) {
    const size_t idx = (start_idx + count) % capacity;
    T* key = table + idx * 2;
    T* value = key + 1;
    if (TryGetOrInsert<T, T>(key, value, size, hash, out)) { return true; }
  }
  return false;
}

template<typename T>
__global__ void EncodeGpu(const size_t capacity, T* table, T* size, const int64_t n, const T* hash,
                          T* out) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    bool success = GetOrInsertOne<T>(capacity, table, size, hash[i], out + i);
    assert(success);
  }
}

}  // namespace

template<typename T>
struct CategoricalOrdinalEncodeKernelUtil<DeviceType::kCUDA, T> {
  static void Encode(ep::Stream* stream, int64_t capacity, T* table, T* size, int64_t n,
                     const T* hash, T* out) {
    EncodeGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(capacity, table, size, n, hash, out);
  }
};

#define INSTANTIATE_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_CUDA(type_cpp, type_proto) \
  template struct CategoricalOrdinalEncodeKernelUtil<DeviceType::kCUDA, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_CUDA, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_CUDA

}  // namespace oneflow
