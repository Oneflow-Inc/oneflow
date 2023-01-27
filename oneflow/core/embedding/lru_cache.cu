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

// Inspired by https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/gpu_cache/src/nv_gpu_cache.cu

#include "oneflow/core/embedding/lru_cache.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/embedding/hash_functions.cuh"
#include <new>
#include <cuda.h>

#if CUDA_VERSION >= 11000 && ((!defined(__CUDA_ARCH__)) || (__CUDA_ARCH__ >= 700)) \
    && !(defined(__clang__) && defined(__CUDA__))
#include <cuda/std/semaphore>
#endif

namespace oneflow {

namespace embedding {

namespace {

constexpr int kWarpSize = 32;
constexpr int kNumWarpPerBlock = 4;
constexpr int kBlockSize = kNumWarpPerBlock * kWarpSize;
constexpr uint32_t kFullMask = 0xFFFFFFFFU;

ep::CudaLaunchConfig GetLaunchConfig(uint32_t n_keys) {
  return ep::CudaLaunchConfig((n_keys + kNumWarpPerBlock - 1) / kNumWarpPerBlock,
                              kWarpSize * kNumWarpPerBlock, 0);
}

struct ThreadContext {
  __device__ ThreadContext() {
    const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    global_warp_id = global_thread_id / kWarpSize;
    warp_id_in_block = global_warp_id % kNumWarpPerBlock;  // NOLINT
    num_warps = gridDim.x * kNumWarpPerBlock;              // NOLINT
    lane_id = global_thread_id % kWarpSize;
  }

  uint32_t global_warp_id;
  uint32_t warp_id_in_block;
  uint32_t num_warps;
  uint32_t lane_id;
};

class WarpMutexAtomicImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WarpMutexAtomicImpl);
  __device__ WarpMutexAtomicImpl() : flag_(0) {}
  __device__ ~WarpMutexAtomicImpl() = default;

  __device__ void Lock(const ThreadContext& thread_ctx) {
    if (thread_ctx.lane_id == 0) {
      while (atomicCAS(&flag_, 0, 1) != 0)
        ;
    }
    __threadfence();
    __syncwarp();
  }

  __device__ void Unlock(const ThreadContext& thread_ctx) {
    __syncwarp();
    __threadfence();
    if (thread_ctx.lane_id == 0) { atomicExch(&flag_, 0); }
  }

 private:
  int32_t flag_;
};

#if CUDA_VERSION >= 11000 && ((!defined(__CUDA_ARCH__)) || (__CUDA_ARCH__ >= 700)) \
    && !(defined(__clang__) && defined(__CUDA__))

class WarpMutexSemaphoreImpl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WarpMutexSemaphoreImpl);
  __device__ WarpMutexSemaphoreImpl() : semaphore_(1) {}
  __device__ ~WarpMutexSemaphoreImpl() = default;

  __device__ void Lock(const ThreadContext& thread_ctx) {
    if (thread_ctx.lane_id == 0) { semaphore_.acquire(); }
    __syncwarp();
  }

  __device__ void Unlock(const ThreadContext& thread_ctx) {
    __syncwarp();
    if (thread_ctx.lane_id == 0) { semaphore_.release(); }
  }

 private:
  cuda::binary_semaphore<cuda::thread_scope_device> semaphore_;
};

#endif

template<typename Key, typename Elem>
struct LruCacheContext {
  Key* keys;
  Elem* lines;
  uint8_t* ages;
  void* mutex;
  uint64_t n_set;
  uint32_t line_size;
  CacheOptions::MemoryKind value_memory_kind;
};

__global__ void InitCacheSetMutex(uint32_t n_set, void* mutex) {
#if CUDA_VERSION >= 11000 && __CUDA_ARCH__ >= 700 && !(defined(__clang__) && defined(__CUDA__))
  using WarpMutex = WarpMutexSemaphoreImpl;
#else
  using WarpMutex = WarpMutexAtomicImpl;
#endif  // CUDA_VERSION >= 11000 && __CUDA_ARCH__ >= 700 && !(defined(__clang__) &&
        // defined(__CUDA__))
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_set) { new (reinterpret_cast<WarpMutex*>(mutex) + idx) WarpMutex; }
}

template<typename Key, typename Elem>
void ClearLruCacheContext(LruCacheContext<Key, Elem>* ctx) {
  OF_CUDA_CHECK(cudaMemset(ctx->keys, 0, ctx->n_set * kWarpSize * sizeof(Key)));
  OF_CUDA_CHECK(cudaMemset(ctx->ages, 0, ctx->n_set * kWarpSize * sizeof(uint8_t)));
  InitCacheSetMutex<<<(ctx->n_set - 1 + 256) / 256, 256>>>(ctx->n_set, ctx->mutex);
}

template<typename Key, typename Elem>
void InitLruCacheContext(const CacheOptions& options, LruCacheContext<Key, Elem>* ctx) {
  const size_t keys_size_per_set = kWarpSize * sizeof(Key);
  const uint32_t line_size = options.value_size / sizeof(Elem);
  const size_t lines_size_per_set = kWarpSize * line_size * sizeof(Elem);
  const size_t ages_size_per_set = kWarpSize * sizeof(uint8_t);
  int device = 0;
  OF_CUDA_CHECK(cudaGetDevice(&device));
  int major = 0;
  OF_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  size_t mutex_size_per_set = 0;
#if CUDA_VERSION >= 11000 && !(defined(__clang__) && defined(__CUDA__))
  if (major >= 7) {
#if !defined(__CUDA_ARCH__)
    mutex_size_per_set = sizeof(WarpMutexSemaphoreImpl);
#else
    UNIMPLEMENTED();
#endif
  } else {
    mutex_size_per_set = sizeof(WarpMutexAtomicImpl);
  }
#else
  mutex_size_per_set = sizeof(WarpMutexAtomicImpl);
#endif  // CUDA_VERSION >= 11000 && !(defined(__clang__) && defined(__CUDA__))
  const size_t n_set = (options.capacity - 1 + kWarpSize) / kWarpSize;
  CHECK_GT(n_set, 0);
  ctx->n_set = n_set;
  ctx->line_size = line_size;
  const size_t keys_size = n_set * keys_size_per_set;
  OF_CUDA_CHECK(cudaMalloc(&(ctx->keys), keys_size));
  const size_t lines_size = n_set * lines_size_per_set;
  if (options.value_memory_kind == CacheOptions::MemoryKind::kDevice) {
    OF_CUDA_CHECK(cudaMalloc(&(ctx->lines), lines_size));
  } else if (options.value_memory_kind == CacheOptions::MemoryKind::kHost) {
    if (ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_NUMA_AWARE_ALLOCATION", false)) {
      OF_CUDA_CHECK(cudaMallocHost(&(ctx->lines), lines_size));
    } else {
      OF_CUDA_CHECK(
          NumaAwareCudaMallocHost(device, reinterpret_cast<void**>(&ctx->lines), lines_size));
    }
  } else {
    UNIMPLEMENTED();
  }
  ctx->value_memory_kind = options.value_memory_kind;
  const size_t ages_size = n_set * ages_size_per_set;
  OF_CUDA_CHECK(cudaMalloc(&(ctx->ages), ages_size));
  const size_t mutex_size = n_set * mutex_size_per_set;
  OF_CUDA_CHECK(cudaMalloc(&(ctx->mutex), mutex_size));

  ClearLruCacheContext(ctx);
}

template<typename Key, typename Elem>
void DestroyLruCacheContext(LruCacheContext<Key, Elem>* ctx) {
  OF_CUDA_CHECK(cudaFree(ctx->keys));
  if (ctx->value_memory_kind == CacheOptions::MemoryKind::kDevice) {
    OF_CUDA_CHECK(cudaFree(ctx->lines));
  } else if (ctx->value_memory_kind == CacheOptions::MemoryKind::kHost) {
    OF_CUDA_CHECK(cudaFreeHost(ctx->lines));
  } else {
    UNIMPLEMENTED();
  }
  OF_CUDA_CHECK(cudaFree(ctx->ages));
  OF_CUDA_CHECK(cudaFree(ctx->mutex));
}

template<typename Key, typename Elem>
struct SetContext {
#if CUDA_VERSION >= 11000 && __CUDA_ARCH__ >= 700 && !(defined(__clang__) && defined(__CUDA__))
  using WarpMutex = WarpMutexSemaphoreImpl;
#else
  using WarpMutex = WarpMutexAtomicImpl;
#endif  // CUDA_VERSION >= 11000 && __CUDA_ARCH__ >= 700 && !(defined(__clang__) &&
        // defined(__CUDA__))
  __device__ SetContext(const LruCacheContext<Key, Elem>& ctx, uint32_t set_id)
      : keys(ctx.keys + set_id * kWarpSize),
        mutex(reinterpret_cast<WarpMutex*>(ctx.mutex) + set_id),
        ages(ctx.ages + set_id * kWarpSize),
        lines(ctx.lines + static_cast<size_t>(set_id) * kWarpSize * ctx.line_size) {}

  __device__ int Lookup(const ThreadContext& thread_ctx, Key key) {
    const Key lane_key = keys[thread_ctx.lane_id];
    const int lane_age = ages[thread_ctx.lane_id];
    const bool lane_hit = (lane_key == key && lane_age != 0);
    const unsigned hit_mask = __ballot_sync(kFullMask, lane_hit);
    if (hit_mask != 0) {
      return __ffs(static_cast<int>(hit_mask)) - 1;
    } else {
      return -1;
    }
  }

  __device__ void Read(const LruCacheContext<Key, Elem>& cache_ctx, const ThreadContext& thread_ctx,
                       int way, Elem* line) {
    const Elem* from_line = lines + way * cache_ctx.line_size;
    for (int i = thread_ctx.lane_id; i < cache_ctx.line_size; i += kWarpSize) {
      line[i] = from_line[i];
    }
  }

  __device__ int InsertWithoutEvicting(const LruCacheContext<Key, Elem>& cache_ctx,
                                       const ThreadContext& thread_ctx, Key key) {
    int insert_way = -1;
    const Key lane_key = keys[thread_ctx.lane_id];
    int lane_age = ages[thread_ctx.lane_id];
    const unsigned hit_mask = __ballot_sync(kFullMask, lane_key == key && lane_age != 0);
    if (hit_mask != 0) {
      insert_way = __ffs(static_cast<int>(hit_mask)) - 1;
      const int insert_way_age = __shfl_sync(kFullMask, lane_age, insert_way);
      if (lane_age > insert_way_age) {
        lane_age -= 1;
      } else if (thread_ctx.lane_id == insert_way) {
        lane_age = kWarpSize;
      }
      __syncwarp();
    }
    if (insert_way == -1) {
      const unsigned valid_mask = __ballot_sync(kFullMask, lane_age != 0);
      if (valid_mask != kFullMask) {
        insert_way = __popc(static_cast<int>(valid_mask));
        if (lane_age > 0) {
          lane_age -= 1;
        } else if (thread_ctx.lane_id == insert_way) {
          lane_age = kWarpSize;
          keys[insert_way] = key;
        }
        __syncwarp();
      }
    }
    if (insert_way != -1) { ages[thread_ctx.lane_id] = lane_age; }
    return insert_way;
  }

  __device__ void Evict(const LruCacheContext<Key, Elem>& cache_ctx,
                        const ThreadContext& thread_ctx, Key key, int* way, Key* evicted_key) {
    const Key lane_key = keys[thread_ctx.lane_id];
    int lane_age = ages[thread_ctx.lane_id];
    const int insert_way = __ffs(__ballot_sync(kFullMask, lane_age == 1)) - 1;
    *evicted_key = __shfl_sync(kFullMask, lane_key, insert_way);
    if (thread_ctx.lane_id == insert_way) {
      keys[insert_way] = key;
      lane_age = kWarpSize;
    } else if (lane_age > 1) {
      lane_age -= 1;
    }
    __syncwarp();
    ages[thread_ctx.lane_id] = lane_age;
    *way = insert_way;
  }

  __device__ void Write(const LruCacheContext<Key, Elem>& cache_ctx,
                        const ThreadContext& thread_ctx, int way, const Elem* line) {
    Elem* to_line = lines + way * cache_ctx.line_size;
    for (int i = thread_ctx.lane_id; i < cache_ctx.line_size; i += kWarpSize) {
      to_line[i] = line[i];
    }
  }

  __device__ void Lock(const ThreadContext& thread_ctx) { mutex->Lock(thread_ctx); }

  __device__ void Unlock(const ThreadContext& thread_ctx) { mutex->Unlock(thread_ctx); }

  Key* keys;
  Elem* lines;
  uint8_t* ages;
  WarpMutex* mutex;
};

template<typename Key, typename Elem, bool test_only>
__global__ void GetKernel(LruCacheContext<Key, Elem> cache_ctx, uint32_t num_keys, const Key* keys,
                          Elem* values, uint32_t* n_missing_keys, Key* missing_keys,
                          uint32_t* missing_indices) {
  ThreadContext thread_ctx{};
  __shared__ Key block_keys[kNumWarpPerBlock][kWarpSize];
  __shared__ size_t block_set_ids[kNumWarpPerBlock][kWarpSize];
  for (uint32_t batch_offset = thread_ctx.global_warp_id * kWarpSize; batch_offset < num_keys;
       batch_offset += thread_ctx.num_warps * kWarpSize) {
    const uint32_t n_batch_keys = min(kWarpSize, num_keys - batch_offset);
    if (thread_ctx.lane_id < n_batch_keys) {
      const Key key = keys[batch_offset + thread_ctx.lane_id];
      const size_t hash = LruCacheHash()(key);
      const uint32_t set_id = hash % cache_ctx.n_set;
      block_keys[thread_ctx.warp_id_in_block][thread_ctx.lane_id] = key;
      block_set_ids[thread_ctx.warp_id_in_block][thread_ctx.lane_id] = set_id;
    }
    __syncwarp();
    uint32_t n_warp_missing = 0;
    Key warp_missing_key = 0;
    uint32_t warp_missing_index = 0;
    for (uint32_t i = 0; i < n_batch_keys; ++i) {
      const uint32_t key_idx = batch_offset + i;
      const Key key = block_keys[thread_ctx.warp_id_in_block][i];
      const size_t set_id = block_set_ids[thread_ctx.warp_id_in_block][i];
      SetContext<Key, Elem> set_ctx(cache_ctx, set_id);
      const int way = set_ctx.Lookup(thread_ctx, key);
      if (way < 0) {
        if (thread_ctx.lane_id == n_warp_missing) {
          warp_missing_key = key;
          warp_missing_index = key_idx;
        }
        __syncwarp();
        n_warp_missing += 1;
      } else if (!test_only) {
        set_ctx.Read(cache_ctx, thread_ctx, way, values + key_idx * cache_ctx.line_size);
      }
    }
    if (n_warp_missing > 0) {
      uint32_t base_missing_idx = 0;
      if (thread_ctx.lane_id == 0) { base_missing_idx = atomicAdd(n_missing_keys, n_warp_missing); }
      __syncwarp();
      base_missing_idx = __shfl_sync(kFullMask, base_missing_idx, 0);
      if (thread_ctx.lane_id < n_warp_missing) {
        missing_keys[base_missing_idx + thread_ctx.lane_id] = warp_missing_key;
        missing_indices[base_missing_idx + thread_ctx.lane_id] = warp_missing_index;
      }
      __syncwarp();
    }
    __syncwarp();
  }
}

template<typename Key, typename Elem>
__global__ void PutWithoutEvictingKernel(LruCacheContext<Key, Elem> cache_ctx, uint32_t num_keys,
                                         const Key* keys, const Elem* values, uint32_t* n_missing,
                                         Key* missing_keys, uint32_t* missing_indices) {
  ThreadContext thread_ctx{};
  __shared__ Key block_keys[kNumWarpPerBlock][kWarpSize];
  __shared__ size_t block_set_ids[kNumWarpPerBlock][kWarpSize];
  for (uint32_t batch_offset = thread_ctx.global_warp_id * kWarpSize; batch_offset < num_keys;
       batch_offset += thread_ctx.num_warps * kWarpSize) {
    const uint32_t n_batch_keys = min(kWarpSize, num_keys - batch_offset);
    if (thread_ctx.lane_id < n_batch_keys) {
      const Key key = keys[batch_offset + thread_ctx.lane_id];
      const size_t hash = LruCacheHash()(key);
      const uint32_t set_id = hash % cache_ctx.n_set;
      block_keys[thread_ctx.warp_id_in_block][thread_ctx.lane_id] = key;
      block_set_ids[thread_ctx.warp_id_in_block][thread_ctx.lane_id] = set_id;
    }
    __syncwarp();
    uint32_t n_warp_missing = 0;
    Key warp_missing_key = 0;
    uint32_t warp_missing_index = 0;
    for (uint32_t i = 0; i < n_batch_keys; ++i) {
      const uint32_t key_idx = batch_offset + i;
      const Key key = block_keys[thread_ctx.warp_id_in_block][i];
      const size_t set_id = block_set_ids[thread_ctx.warp_id_in_block][i];
      SetContext<Key, Elem> set_ctx(cache_ctx, set_id);
      set_ctx.Lock(thread_ctx);
      Key evicted_key = 0;
      const int insert_way = set_ctx.InsertWithoutEvicting(cache_ctx, thread_ctx, key);
      if (insert_way >= 0) {
        set_ctx.Write(cache_ctx, thread_ctx, insert_way, values + cache_ctx.line_size * key_idx);
      } else {
        if (thread_ctx.lane_id == n_warp_missing) {
          warp_missing_key = key;
          warp_missing_index = key_idx;
        }
        __syncwarp();
        n_warp_missing += 1;
      }
      set_ctx.Unlock(thread_ctx);
    }
    if (n_warp_missing > 0) {
      uint32_t base_missing_idx = 0;
      if (thread_ctx.lane_id == 0) { base_missing_idx = atomicAdd(n_missing, n_warp_missing); }
      __syncwarp();
      base_missing_idx = __shfl_sync(kFullMask, base_missing_idx, 0);
      if (thread_ctx.lane_id < n_warp_missing) {
        missing_keys[base_missing_idx + thread_ctx.lane_id] = warp_missing_key;
        missing_indices[base_missing_idx + thread_ctx.lane_id] = warp_missing_index;
      }
      __syncwarp();
    }
  }
}

template<typename Key, typename Elem>
__global__ void EvictKernel(LruCacheContext<Key, Elem> cache_ctx, const Key* keys,
                            const uint32_t* indices, const Elem* values, const uint32_t* n_evict,
                            Key* evicted_keys, Elem* evicted_values) {
  ThreadContext thread_ctx{};
  uint32_t num_evict = *n_evict;
  __shared__ Key block_keys[kNumWarpPerBlock][kWarpSize];
  __shared__ size_t block_set_ids[kNumWarpPerBlock][kWarpSize];
  for (uint32_t batch_offset = thread_ctx.global_warp_id * kWarpSize; batch_offset < num_evict;
       batch_offset += thread_ctx.num_warps * kWarpSize) {
    const uint32_t n_batch_keys = min(kWarpSize, num_evict - batch_offset);
    if (thread_ctx.lane_id < n_batch_keys) {
      const Key key = keys[batch_offset + thread_ctx.lane_id];
      const size_t hash = LruCacheHash()(key);
      const uint32_t set_id = hash % cache_ctx.n_set;
      block_keys[thread_ctx.warp_id_in_block][thread_ctx.lane_id] = key;
      block_set_ids[thread_ctx.warp_id_in_block][thread_ctx.lane_id] = set_id;
    }
    __syncwarp();
    for (uint32_t i = 0; i < n_batch_keys; ++i) {
      const uint32_t key_idx = batch_offset + i;
      const Key key = block_keys[thread_ctx.warp_id_in_block][i];
      const uint32_t set_id = block_set_ids[thread_ctx.warp_id_in_block][i];
      SetContext<Key, Elem> set_ctx(cache_ctx, set_id);
      set_ctx.Lock(thread_ctx);
      int evicted_way = -1;
      Key evicted_key = 0;
      set_ctx.Evict(cache_ctx, thread_ctx, key, &evicted_way, &evicted_key);
      if (thread_ctx.lane_id == 0) { evicted_keys[key_idx] = evicted_key; }
      __syncwarp();
      set_ctx.Read(cache_ctx, thread_ctx, evicted_way,
                   evicted_values + cache_ctx.line_size * key_idx);
      set_ctx.Write(cache_ctx, thread_ctx, evicted_way,
                    values + cache_ctx.line_size * indices[key_idx]);
      set_ctx.Unlock(thread_ctx);
    }
  }
}

template<typename Key, typename Elem>
__global__ void DumpKernel(LruCacheContext<Key, Elem> cache_ctx, size_t start_key_index,
                           size_t end_key_index, uint32_t* n_dumped, Key* keys, Elem* values) {
  ThreadContext thread_ctx{};
  __shared__ Key warp_keys[kNumWarpPerBlock][kWarpSize];
  __shared__ uint8_t warp_ages[kNumWarpPerBlock][kWarpSize];
  for (uint32_t warp_start_key_index = start_key_index + thread_ctx.global_warp_id * kWarpSize;
       warp_start_key_index < end_key_index;
       warp_start_key_index += thread_ctx.num_warps * kWarpSize) {
    Key lane_key = 0;
    uint8_t lane_age = 0;
    if (warp_start_key_index + thread_ctx.lane_id < end_key_index) {
      lane_key = cache_ctx.keys[warp_start_key_index + thread_ctx.lane_id];
      lane_age = cache_ctx.ages[warp_start_key_index + thread_ctx.lane_id];
    }
    __syncwarp();
    warp_keys[thread_ctx.warp_id_in_block][thread_ctx.lane_id] = lane_key;
    warp_ages[thread_ctx.warp_id_in_block][thread_ctx.lane_id] = lane_age;
    const int key_count = __popc(__ballot_sync(kFullMask, lane_age != 0));
    if (key_count == 0) { continue; }
    uint32_t offset = 0;
    if (thread_ctx.lane_id == 0) { offset = atomicAdd(n_dumped, key_count); }
    offset = __shfl_sync(kFullMask, offset, 0);
    __syncwarp();
    for (uint32_t i = 0; i < kWarpSize; ++i) {
      const Key key = warp_keys[thread_ctx.warp_id_in_block][i];
      const Key age = warp_ages[thread_ctx.warp_id_in_block][i];
      if (age == 0) { continue; }
      if (thread_ctx.lane_id == 0) { keys[offset] = key; }
      __syncwarp();
      for (uint32_t j = thread_ctx.lane_id; j < cache_ctx.line_size; j += kWarpSize) {
        values[offset * cache_ctx.line_size + j] =
            cache_ctx
                .lines[static_cast<size_t>(warp_start_key_index + i) * cache_ctx.line_size + j];
      }
      __syncwarp();
      offset += 1;
    }
  }
}

template<typename Key, typename Elem>
class LruCache : public Cache {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LruCache);
  explicit LruCache(const CacheOptions& options)
      : device_index_{},
        max_query_length_(0),
        query_indices_buffer_(nullptr),
        query_keys_buffer_(nullptr),
        value_type_(options.value_type) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    InitLruCacheContext(options, &ctx_);
  }
  ~LruCache() override {
    CudaCurrentDeviceGuard guard(device_index_);
    if (max_query_length_ != 0) {
      OF_CUDA_CHECK(cudaFree(query_indices_buffer_));
      OF_CUDA_CHECK(cudaFree(query_keys_buffer_));
    }
    DestroyLruCacheContext(&ctx_);
  }

  uint32_t KeySize() const override { return sizeof(Key); }
  uint32_t ValueSize() const override { return sizeof(Elem) * ctx_.line_size; }
  DataType ValueType() const override { return value_type_; }
  uint64_t Capacity() const override { return ctx_.n_set * kWarpSize; }
  uint32_t MaxQueryLength() const override { return max_query_length_; }

  void ReserveQueryLength(uint32_t query_length) override {
    CudaCurrentDeviceGuard guard(device_index_);
    if (query_length < max_query_length_) { return; }
    if (max_query_length_ != 0) {
      OF_CUDA_CHECK(cudaFree(query_indices_buffer_));
      OF_CUDA_CHECK(cudaFree(query_keys_buffer_));
    }
    OF_CUDA_CHECK(cudaMalloc(&query_indices_buffer_, query_length * sizeof(uint32_t)));
    OF_CUDA_CHECK(cudaMalloc(&query_keys_buffer_, query_length * sizeof(Key)));
    max_query_length_ = query_length;
  }

  CacheOptions::Policy Policy() const override { return CacheOptions::Policy::kLRU; }

  void Test(ep::Stream* stream, uint32_t n_keys, const void* keys, uint32_t* n_missing,
            void* missing_keys, uint32_t* missing_indices) override {
    CHECK_LE(n_keys, max_query_length_);
    auto cuda_stream = stream->As<ep::CudaStream>();
    OF_CUDA_CHECK(cudaMemsetAsync(n_missing, 0, sizeof(uint32_t), cuda_stream->cuda_stream()));
    if (n_keys == 0) { return; }
    cuda_stream->LaunchKernel(GetKernel<Key, Elem, true>, GetLaunchConfig(n_keys), ctx_, n_keys,
                              static_cast<const Key*>(keys), nullptr, n_missing,
                              static_cast<Key*>(missing_keys), missing_indices);
  }

  using Cache::Get;
  void Get(ep::Stream* stream, uint32_t n_keys, const void* keys, void* values, uint32_t* n_missing,
           void* missing_keys, uint32_t* missing_indices) override {
    CHECK_LE(n_keys, max_query_length_);
    auto cuda_stream = stream->As<ep::CudaStream>();
    OF_CUDA_CHECK(cudaMemsetAsync(n_missing, 0, sizeof(uint32_t), cuda_stream->cuda_stream()));
    if (n_keys == 0) { return; }
    cuda_stream->LaunchKernel(GetKernel<Key, Elem, false>, GetLaunchConfig(n_keys), ctx_, n_keys,
                              static_cast<const Key*>(keys), static_cast<Elem*>(values), n_missing,
                              static_cast<Key*>(missing_keys), missing_indices);
  }

  void Put(ep::Stream* stream, uint32_t n_keys, const void* keys, const void* values,
           uint32_t* n_evicted, void* evicted_keys, void* evicted_values) override {
    CHECK_LE(n_keys, max_query_length_);
    auto cuda_stream = stream->As<ep::CudaStream>();
    OF_CUDA_CHECK(cudaMemsetAsync(n_evicted, 0, sizeof(uint32_t), cuda_stream->cuda_stream()));
    if (n_keys == 0) { return; }
    cuda_stream->LaunchKernel(PutWithoutEvictingKernel<Key, Elem>, GetLaunchConfig(n_keys), ctx_,
                              n_keys, static_cast<const Key*>(keys),
                              static_cast<const Elem*>(values), n_evicted, query_keys_buffer_,
                              query_indices_buffer_);
    cuda_stream->LaunchKernel(EvictKernel<Key, Elem>, GetLaunchConfig(n_keys), ctx_,
                              query_keys_buffer_, query_indices_buffer_,
                              static_cast<const Elem*>(values), n_evicted,
                              static_cast<Key*>(evicted_keys), static_cast<Elem*>(evicted_values));
  }

  void Dump(ep::Stream* stream, uint64_t start_key_index, uint64_t end_key_index,
            uint32_t* n_dumped, void* keys, void* values) override {
    auto cuda_stream = stream->As<ep::CudaStream>();
    OF_CUDA_CHECK(cudaMemsetAsync(n_dumped, 0, sizeof(uint32_t), cuda_stream->cuda_stream()));
    const uint64_t max_dump_keys = end_key_index - start_key_index;
    cuda_stream->LaunchKernel(
        DumpKernel<Key, Elem>,
        ep::CudaLaunchConfig((max_dump_keys + kNumWarpPerBlock - 1) / kNumWarpPerBlock, kBlockSize,
                             0),
        ctx_, start_key_index, end_key_index, n_dumped, static_cast<Key*>(keys),
        static_cast<Elem*>(values));
  }

  void ClearDirtyFlags() override {
    // do nothing.
    return;
  }

  void Clear() override { ClearLruCacheContext<Key, Elem>(&ctx_); }

 private:
  int device_index_;
  uint32_t max_query_length_;
  LruCacheContext<Key, Elem> ctx_;
  uint32_t* query_indices_buffer_;
  Key* query_keys_buffer_;
  DataType value_type_;
};

template<typename Key>
std::unique_ptr<Cache> DispatchValueType(const CacheOptions& options) {
  if (options.value_size % sizeof(ulonglong2) == 0) {
    return std::unique_ptr<Cache>(new LruCache<Key, ulonglong2>(options));
  } else if (options.value_size % sizeof(uint64_t) == 0) {
    return std::unique_ptr<Cache>(new LruCache<Key, uint64_t>(options));
  } else if (options.value_size % sizeof(uint32_t) == 0) {
    return std::unique_ptr<Cache>(new LruCache<Key, uint32_t>(options));
  } else if (options.value_size % sizeof(uint16_t) == 0) {
    return std::unique_ptr<Cache>(new LruCache<Key, uint16_t>(options));
  } else {
    return std::unique_ptr<Cache>(new LruCache<Key, uint8_t>(options));
  }
}

std::unique_ptr<Cache> DispatchKeyType(const CacheOptions& options) {
  if (options.key_size == sizeof(uint32_t)) {
    return DispatchValueType<uint32_t>(options);
  } else if (options.key_size == sizeof(uint64_t)) {
    return DispatchValueType<uint64_t>(options);
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

}  // namespace

std::unique_ptr<Cache> NewLruCache(const CacheOptions& options) { return DispatchKeyType(options); }

}  // namespace embedding

}  // namespace oneflow
