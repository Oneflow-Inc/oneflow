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
#ifndef ONEFLOW_CORE_EMBEDDING_CUDA_CACHE_H_
#define ONEFLOW_CORE_EMBEDDING_CUDA_CACHE_H_

#include <cub/cub.cuh>
#include <cassert>

namespace oneflow {

namespace embedding {

namespace {

constexpr int kWarpSize = 32;
constexpr uint32_t kFullMask = 0xFFFFFFFFU;

__device__ __forceinline__ int32_t FirstLane(int pred) {
  return __ffs(static_cast<int32_t>(__ballot_sync(kFullMask, pred))) - 1;
}

template<typename T>
class BatchIO final {
 public:
  __device__ __forceinline__ explicit BatchIO(int32_t lane, int32_t size, const T* ptr)
      : lane_(lane), size_(size) {
    Load(ptr);
  };

  __device__ __forceinline__ explicit BatchIO(int32_t lane, int32_t size, T val)
      : lane_(lane), size_(size), val_(val){};

  __device__ __forceinline__ explicit BatchIO(int32_t lane, int32_t size)
      : lane_(lane), size_(size){};

  __device__ __forceinline__ void Load(const T* ptr) {
    if (lane_ < size_) { val_ = ptr[lane_]; }
  }

  __device__ __forceinline__ void Store(T* ptr) {
    if (lane_ < size_) { ptr[lane_] = val_; }
  }

  __device__ __forceinline__ void Set(int32_t lane, T val) {
    if (lane_ == lane) { val_ = val; }
  }

  __device__ __forceinline__ T Get(int32_t lane) { return __shfl_sync(kFullMask, val_, lane); }

 private:
  T val_;
  int32_t lane_;
  int32_t size_;
};

enum Status { kInvalid = 0, kLoaded, kModified };

class LFUQueue final {
 public:
  __device__ __forceinline__ LFUQueue(int32_t lane, int8_t* ptr) : lane_(lane) { Reset(ptr); }

  __device__ __forceinline__ LFUQueue(int32_t lane) : lane_(lane) { Reset(); }

  __device__ __forceinline__ ~LFUQueue() { Reset(); }

  __device__ __forceinline__ void Load() {
    if (status_ == Status::kInvalid) {
      val_ = static_cast<int32_t>(ptr_[lane_]);
      status_ = Status::kLoaded;
    }
  }

  __device__ __forceinline__ void Reset(int8_t* ptr) {
    if (status_ == Status::kModified) { ptr_[lane_] = static_cast<int8_t>(val_); }
    ptr_ = ptr;
    status_ = Status::kInvalid;
  }

  __device__ __forceinline__ void Reset() { Reset(nullptr); }

  __device__ __forceinline__ void Push(int32_t val) {
    Load();
    const int32_t prev_lane = FirstLane(val_ == val);
    // LRU
    if (prev_lane == -1 || lane_ >= prev_lane) { val_ = __shfl_down_sync(__activemask(), val_, 1); }
    if (lane_ == 31) { val_ = val; }
    status_ = Status::kModified;
  }

  __device__ __forceinline__ int32_t Front() {
    Load();
    return __shfl_sync(kFullMask, val_, 0);
  }

 private:
  int32_t lane_;
  int32_t val_;
  int8_t* ptr_;
  Status status_;
};

template<typename Key, typename Elem, typename Idx>
class CacheSet final {
 public:
  static const uint64_t kInvalidId = (~0x0ULL);
  __device__ __forceinline__ CacheSet(int32_t lane, int32_t line_size)
      : lane_(lane),
        line_size_(line_size),
        id_(kInvalidId),
        key_ptr_(nullptr),
        lines_ptr_(nullptr),
        valid_bits_ptr_(nullptr),
        lfu_queue_(lane),
        mutex_(nullptr) {
    mask_ = 1U << lane_;
  }

  __device__ __forceinline__ ~CacheSet() { Reset(); }

  __device__ __forceinline__ uint64_t id() const { return id_; }

  __device__ __forceinline__ void Reset(uint64_t id, Key* key, Elem* lines, uint32_t* valid_bits,
                                        int8_t* lfu_queue_buffer_ptr, int32_t* mutex) {
    if (id_ != kInvalidId && lane_ == 0) { *valid_bits_ptr_ = valid_bits_; }
    id_ = id;
    key_ptr_ = key;
    lines_ptr_ = lines;
    valid_bits_ptr_ = valid_bits;
    if (id_ != kInvalidId) { valid_bits_ = *valid_bits; }
    lfu_queue_.Reset(lfu_queue_buffer_ptr);
    mutex_ = mutex;
  }

  __device__ __forceinline__ void Reset() {
    Reset(kInvalidId, nullptr, nullptr, nullptr, nullptr, nullptr);
  }

  __device__ __forceinline__ int32_t LookupWay(Key key, uint64_t hash) {
    const bool hit = ((valid_bits_ & mask_) != 0) && (key_ptr_[lane_] == key);
    return FirstLane(hit);
  }

  template<bool key_only>
  __device__ __forceinline__ int32_t Lookup(Key key, uint64_t hash, Elem* line) {
    const int32_t way = LookupWay(key, hash);
    if (!key_only) {
      if (way >= 0) { ReadLine(line, way); }
    }
    return way;
  }

  __device__ __forceinline__ void Update(Key key, uint64_t hash, const Elem* line, Idx* n_evicted,
                                         Key* evicted_keys, Elem* evicted_lines) {
    int32_t recent_way = -1;
    int32_t way = LookupWay(key, hash);
    if (way >= 0) {
      WriteLine(line, way);
      recent_way = way;
    } else if (valid_bits_ != kFullMask) {
      const int32_t invalid_way = __ffs(~valid_bits_) - 1;
      SetWay(invalid_way, key, hash, line);
      recent_way = invalid_way;
    } else {
      __shared__ Idx evicted_idx;
      if (lane_ == 0) { evicted_idx = atomicAdd(n_evicted, 1); }
      __syncwarp();
      int32_t evict_way = lfu_queue_.Front();
      EvictWay(evict_way, key, hash, line, evicted_keys + evicted_idx,
               evicted_lines + evicted_idx * line_size_);
      recent_way = evict_way;
    }
    if (recent_way >= 0) { lfu_queue_.Push(recent_way); }
  }

  __device__ __forceinline__ void ReadLine(Elem* dst, int32_t way) const {
    const Elem* line = lines_ptr_ + way * line_size_;
    for (int32_t i = lane_; i < line_size_; i += 32) { dst[i] = line[i]; }
  }

  __device__ __forceinline__ void WriteLine(const Elem* src, int32_t way) {
    Elem* line = lines_ptr_ + way * line_size_;
    for (int32_t i = lane_; i < line_size_; i += 32) { line[i] = src[i]; }
  }

  __device__ __forceinline__ void SetWay(int32_t way, Key key, uint32_t hash, const Elem* line) {
    if (lane_ == 0) { key_ptr_[way] = key; }
    const uint32_t mask = (1U << way);
    valid_bits_ |= mask;
    WriteLine(line, way);
  }

  __device__ __forceinline__ void EvictWay(int32_t way, Key key, uint32_t hash, const Elem* line,
                                           Key* evict_key, Elem* evict_line) {
    if (lane_ == 0) {
      *evict_key = key_ptr_[way];
      key_ptr_[way] = key;
    }
    Elem* cache_line = lines_ptr_ + way * line_size_;
    for (int32_t i = lane_; i < line_size_; i += 32) {
      evict_line[i] = cache_line[i];
      cache_line[i] = line[i];
    }
  }

 private:
  int32_t lane_;
  int32_t line_size_;
  uint64_t id_;
  uint32_t mask_;
  uint32_t valid_bits_;
  Key* key_ptr_;
  Elem* lines_ptr_;
  uint32_t* valid_bits_ptr_;
  LFUQueue lfu_queue_;
  int32_t* mutex_;
};

template<typename Key, typename Elem, typename Idx>
struct CacheContext {
  Key* keys;
  Elem* lines;
  uint32_t* valid_bits;
  int8_t* lfu_queue_buffer;
  int32_t* mutex;
  uint32_t log2_n_set;
  uint32_t line_size;

  static constexpr uint64_t kAlignSize = 1024;
  static constexpr uint64_t kNWay = 32;
  static constexpr uint32_t kBlockSize = 128;
  static constexpr uint32_t kMaxLog2NWorker = 15;
  static_assert(kBlockSize % kWarpSize == 0, "");
  static constexpr uint32_t kWarpPerBlock = kBlockSize / kWarpSize;

  CacheContext(uint32_t log2_n_set, uint32_t line_size, void* workspace, uint64_t workspace_size)
      : log2_n_set(log2_n_set), line_size(line_size) {
    static_assert(sizeof(uint32_t) * 8 == kNWay, "");
    assert(workspace_size >= GetWorkspaceSize(log2_n_set, line_size));
    const uint64_t n_set = 1ULL << log2_n_set;
    keys = OffsetPtr<Key>(workspace, 0);
    lines = OffsetPtr<Elem>(keys, KeysSize(n_set));
    valid_bits = OffsetPtr<uint32_t>(lines, LinesSize(n_set, line_size));
    lfu_queue_buffer = OffsetPtr<int8_t>(valid_bits, ValidBitsSize(n_set));
    mutex = OffsetPtr<int32_t>(lfu_queue_buffer, LFUQueueBufferSize(n_set));
  }

  __device__ __host__ __forceinline__ static uint64_t GetWorkspaceSize(uint32_t log2_n_set,
                                                                       uint32_t line_size) {
    const uint64_t n_set = 1ULL << log2_n_set;
    return KeysSize(n_set) + LinesSize(n_set, line_size) + ValidBitsSize(n_set)
           + LFUQueueBufferSize(n_set) + MutexSize(n_set);
  }

  template<typename T>
  __device__ __host__ __forceinline__ static T* OffsetPtr(void* ptr, uint64_t size) {
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr) + size);
  }

  __device__ __host__ __forceinline__ static uint64_t KeysSize(uint64_t n_set) {
    return AlignedSize(n_set * kNWay * sizeof(Key));
  }

  __device__ __host__ __forceinline__ static uint64_t LinesSize(uint64_t n_set,
                                                                uint32_t line_size) {
    return AlignedSize(n_set * kNWay * line_size * sizeof(Elem));
  }

  __device__ __host__ __forceinline__ static uint64_t ValidBitsSize(uint64_t n_set) {
    return AlignedSize(n_set * sizeof(uint32_t));
  }

  __device__ __host__ __forceinline__ static uint64_t LFUQueueBufferSize(uint64_t n_set) {
    return AlignedSize(n_set * kNWay * sizeof(int8_t));
  }

  __device__ __host__ __forceinline__ static uint64_t MutexSize(uint64_t n_set) {
    return AlignedSize(n_set * sizeof(int32_t));
  }

  __device__ __host__ __forceinline__ static uint64_t AlignedSize(uint64_t size) {
    return (size + kAlignSize - 1) / kAlignSize * kAlignSize;
  }

  __device__ __forceinline__ void ResetSet(CacheSet<Key, Elem, Idx>& set, uint64_t set_idx) {
    if (set.id() != set_idx) {
      const uint64_t offset = set_idx * 32;
      set.Reset(set_idx, keys + offset, lines + offset * line_size, valid_bits + set_idx,
                lfu_queue_buffer + offset, mutex + set_idx);
    }
  }

  __device__ __forceinline__ void ResetSet(CacheSet<Key, Elem, Idx>& set) { set.Reset(); }

  __device__ __host__ __forceinline__ uint32_t GetLog2WorkerNum() {
    return min(log2_n_set, kMaxLog2NWorker);
  }

  __device__ __host__ __forceinline__ uint32_t GetWorkerNum() { return 1U << GetLog2WorkerNum(); }

  __device__ __host__ __forceinline__ uint32_t GetWorkerId(uint64_t hash) {
    return static_cast<uint32_t>(hash >> (sizeof(uint64_t) * 8 - GetLog2WorkerNum()));
  }

  __device__ __forceinline__ void ResetSetByHash(CacheSet<Key, Elem, Idx>& set, uint64_t hash) {
    const uint64_t set_idx = hash >> (sizeof(uint64_t) * 8 - log2_n_set);
    ResetSet(set, set_idx);
  }

  __device__ __host__ __forceinline__ dim3 GetBlockDim() {
    return dim3(kWarpSize, min(GetWorkerNum(), kWarpPerBlock));
  }

  __device__ __host__ __forceinline__ dim3 GetGridDim() {
    return dim3(max(GetWorkerNum() / kWarpPerBlock, 1));
  }
};

template<typename Key, typename Elem, typename Idx>
__global__ void InitKernel(CacheContext<Key, Elem, Idx> ctx) {
  uint64_t n_set = 1ULL << ctx.log2_n_set;
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_set; i += gridDim.x * blockDim.x) {
    ctx.valid_bits[i] = 0;
    ctx.mutex[i] = 0;
  }
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_set * ctx.kNWay;
       i += gridDim.x * blockDim.x) {
    // LFU
    ctx.lfu_queue_buffer[i] = i % ctx.kNWay;
  }
}

template<typename Key, typename Elem, typename Idx, typename Hash, bool key_only>
__device__ void LookupKernelImpl(CacheContext<Key, Elem, Idx> ctx, uint64_t n_key, const Key* keys,
                                 Elem* lines, Idx* n_miss, Key* miss_keys, Idx* miss_indices) {
  const uint32_t worker_id = blockIdx.x * blockDim.y + threadIdx.y;
  const auto lane = static_cast<int32_t>(threadIdx.x);
  CacheSet<Key, Elem, Idx> set(lane, ctx.line_size);
  for (uint32_t key_id = worker_id * 32; key_id < n_key; key_id += 32 * ctx.GetWorkerNum()) {
    const auto batch_size = min(32, static_cast<int32_t>(n_key - key_id));
    BatchIO<Key> batch_key(lane, batch_size, keys + key_id);
    for (int32_t idx = 0; idx < batch_size; ++idx) {
      const Key key = batch_key.Get(idx);
      const uint64_t hash = Hash()(key);
      const uint64_t set_idx = hash >> (sizeof(uint64_t) * 8 - ctx.log2_n_set);
      ctx.ResetSet(set, set_idx);
      Elem* line = key_only ? nullptr : lines + (key_id + idx) * ctx.line_size;
      const int32_t way = set.Lookup<key_only>(key, hash, line);
      ctx.ResetSet(set);
      if (way < 0 && lane == 0) {
        Idx pos = atomicAdd(n_miss, 1);
        miss_keys[pos] = key;
        if (!key_only) { miss_indices[pos] = key_id + idx; }
      }
      __syncwarp();
    }
  }
}

template<typename Key, typename Elem, typename Idx, typename Hash, bool key_only>
__global__ void LookupKernel(CacheContext<Key, Elem, Idx> ctx, Idx n_key, const Key* keys,
                             Elem* lines, Idx* n_miss, Key* miss_keys, Idx* miss_indices) {
  LookupKernelImpl<Key, Elem, Idx, Hash, key_only>(ctx, n_key, keys, lines, n_miss, miss_keys,
                                                   miss_indices);
}

template<typename Key, typename Elem, typename Idx, typename Hash, bool key_only>
__global__ void LookupKernel(CacheContext<Key, Elem, Idx> ctx, const Idx* n_key, const Key* keys,
                             Elem* lines, Idx* n_miss, Key* miss_keys, Idx* miss_indices) {
  LookupKernelImpl<Key, Elem, Idx, Hash, key_only>(ctx, *n_key, keys, lines, n_miss, miss_keys,
                                                   miss_indices);
}

template<typename Key, typename Elem, typename Idx, typename Hash, int block_size>
__device__ void UpdateKernelImpl(CacheContext<Key, Elem, Idx> ctx, Idx n_key, const Key* keys,
                                 const Elem* lines, Idx* n_evicted, Key* evicted_keys,
                                 Elem* evicted_lines) {
  const uint32_t worker_id = blockIdx.x * blockDim.y + threadIdx.y;
  const auto lane = static_cast<int32_t>(threadIdx.x);
  __shared__ Key shared_keys[block_size / kWarpSize][kWarpSize];
  __shared__ uint64_t shared_hashes[block_size / kWarpSize][kWarpSize];
  CacheSet<Key, Elem, Idx> set(lane, ctx.line_size);
  for (uint32_t key_id = worker_id * 32; key_id < n_key; key_id += ctx.GetWorkerNum() * 32) {
    const auto batch_size = min(32, static_cast<int32_t>(n_key - key_id));
    if (lane < batch_size) {
      const Key key = keys[key_id + lane];
      shared_keys[threadIdx.y][lane] = key;
      shared_hashes[threadIdx.y][lane] = Hash()(key);
    }
    __syncwarp();
    for (int32_t idx = 0; idx < batch_size; ++idx) {
      const uint32_t iter_key_id = key_id + idx;
      const Key key = shared_keys[threadIdx.y][idx];
      const uint64_t hash = shared_hashes[threadIdx.y][idx];
      const uint64_t set_idx = hash >> (sizeof(uint64_t) * 8 - ctx.log2_n_set);
      int32_t* mutex = ctx.mutex + set_idx;
      if (lane == 0) {
        while (atomicCAS(mutex, 0, 1) != 0) {}
      }
      __threadfence();
      __syncwarp();
      ctx.ResetSet(set, set_idx);
      set.Update(key, hash, lines + iter_key_id * ctx.line_size, n_evicted, evicted_keys,
                 evicted_lines);
      ctx.ResetSet(set);
      __threadfence();
      __syncwarp();
      if (lane == 0) { atomicExch(mutex, 0); }
    }
  }
}

template<typename Key, typename Elem, typename Idx, typename Hash, int block_size>
__global__ void UpdateKernel(CacheContext<Key, Elem, Idx> ctx, Idx n_key, const Key* keys,
                             const Elem* lines, Idx* n_evicted, Key* evicted_keys,
                             Elem* evicted_lines) {
  UpdateKernelImpl<Key, Elem, Idx, Hash, block_size>(ctx, n_key, keys, lines, n_evicted,
                                                     evicted_keys, evicted_lines);
}

template<typename Key, typename Elem, typename Idx, typename Hash, int block_size>
__global__ void UpdateKernel(CacheContext<Key, Elem, Idx> ctx, const Idx* n_key, const Key* keys,
                             const Elem* lines, Idx* n_evicted, Key* evicted_keys,
                             Elem* evicted_lines) {
  UpdateKernelImpl<Key, Elem, Idx, Hash, block_size>(ctx, *n_key, keys, lines, n_evicted,
                                                     evicted_keys, evicted_lines);
}

}  // namespace

}  // namespace embedding

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_CUDA_CACHE_H_
