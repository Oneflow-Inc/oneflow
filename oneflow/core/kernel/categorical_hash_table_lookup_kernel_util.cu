#include "oneflow/core/kernel/categorical_hash_table_lookup_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

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

template<typename K, typename V>
__device__ void GetOrInsertOne(const size_t capacity, K* keys, V* values, V* size, const K hash,
                               V* out) {
  if (hash == 0) {
    *out = 0;
    return;
  }
  size_t count = 0;
  bool success = false;
  while (!success) {
    if (count >= capacity) { break; }
    const size_t idx = (static_cast<size_t>(hash) + count) % capacity;
    K* key = keys + idx;
    V* value = values + idx;
    if (count == 0 && *key == hash && *value != 0) {
      *out = *value;
      success = true;
      break;
    }
    if (TryGetOrInsert<K, V>(key, value, size, hash, out)) {
      success = true;
      break;
    } else {
      count += 1;
    }
  }
  assert(success);
}

template<typename K, typename V>
__global__ void GetOrInsertGpu(const size_t capacity, K* keys, V* values, V* size, const int64_t n,
                               const K* hash, V* out) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    GetOrInsertOne<K, V>(capacity, keys, values, size, hash[i], out + i);
  }
}

}  // namespace

template<typename K, typename V>
struct CategoricalHashTableLookupKernelUtil<DeviceType::kGPU, K, V> {
  static void GetOrInsert(DeviceCtx* ctx, int64_t capacity, K* keys, V* values, V* size, int64_t n,
                          const K* hash, V* out) {
    GetOrInsertGpu<K, V>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            capacity, keys, values, size, n, hash, out);
  }
};

#define INSTANTIATE_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_UTIL_GPU(key_type_pair, value_type_pair) \
  template struct CategoricalHashTableLookupKernelUtil<                                           \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(key_type_pair), OF_PP_PAIR_FIRST(value_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_UTIL_GPU,
                                 INDEX_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_UTIL_GPU

}  // namespace oneflow
