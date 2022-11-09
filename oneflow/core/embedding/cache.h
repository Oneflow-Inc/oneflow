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
#ifndef ONEFLOW_CORE_EMBEDDING_CACHE_H_
#define ONEFLOW_CORE_EMBEDDING_CACHE_H_

#include "oneflow/core/embedding/kv_iterator.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace embedding {

struct CacheOptions {
  enum class Policy {
    kLRU,
    kFull,
  };
  enum class MemoryKind {
    kDevice,
    kHost,
  };
  Policy policy = Policy::kLRU;
  MemoryKind value_memory_kind = MemoryKind::kDevice;
  uint64_t capacity{};
  uint32_t key_size{};
  uint32_t value_size{};
  DataType value_type{};
  float load_factor = 0.75;
};

class Cache {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Cache);
  Cache() = default;
  virtual ~Cache() = default;

  virtual uint32_t KeySize() const = 0;
  virtual uint32_t ValueSize() const = 0;
  virtual DataType ValueType() const = 0;
  virtual uint32_t MaxQueryLength() const = 0;
  virtual void ReserveQueryLength(uint32_t query_length) = 0;
  virtual uint64_t Capacity() const = 0;
  virtual uint64_t DumpCapacity() const { return Capacity(); }
  virtual CacheOptions::Policy Policy() const = 0;
  virtual void Test(ep::Stream* stream, uint32_t n_keys, const void* keys, uint32_t* n_missing,
                    void* missing_keys, uint32_t* missing_indices) = 0;
  virtual void Get(ep::Stream* stream, uint32_t n_keys, const void* keys, void* values,
                   uint32_t* n_missing, void* missing_keys, uint32_t* missing_indices) = 0;
  virtual void Get(ep::Stream* stream, uint32_t n_keys, const void* keys, void* values,
                   uint8_t* mask) {
    UNIMPLEMENTED();
  }
  virtual void Put(ep::Stream* stream, uint32_t n_keys, const void* keys, const void* values,
                   uint32_t* n_evicted, void* evicted_keys, void* evicted_values) = 0;
  virtual void FusedHalfUpdatePut(ep::Stream* stream, uint32_t n_keys, const void* keys,
                                  const void* values, const void* update, const float* lr,
                                  float scale, uint32_t* n_evicted, void* evicted_keys,
                                  void* evicted_values) {
    UNIMPLEMENTED();
  }
  virtual void Dump(ep::Stream* stream, uint64_t start_key_index, uint64_t end_key_index,
                    uint32_t* n_dumped, void* keys, void* values) = 0;

  virtual void ClearDirtyFlags() = 0;

  virtual void Clear() = 0;
};

std::unique_ptr<Cache> NewCache(const CacheOptions& options);

}  // namespace embedding

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_CACHE_H_
