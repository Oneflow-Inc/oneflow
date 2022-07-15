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
#ifndef ONEFLOW_CORE_EMBEDDING_KEY_VALUE_STORE_H_
#define ONEFLOW_CORE_EMBEDDING_KEY_VALUE_STORE_H_

#include "oneflow/core/embedding/kv_iterator.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

namespace embedding {

class KeyValueStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeyValueStore);
  KeyValueStore() = default;
  virtual ~KeyValueStore() = default;

  virtual uint32_t KeySize() const = 0;
  virtual uint32_t ValueSize() const = 0;
  virtual uint32_t MaxQueryLength() const = 0;
  virtual void ReserveQueryLength(uint32_t query_length) = 0;

  virtual void Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
                   uint32_t* n_missing, uint32_t* missing_indices) = 0;
  virtual void Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
                   uint8_t* mask) {
    UNIMPLEMENTED();
  }
  virtual void Put(ep::Stream* stream, uint32_t num_keys, const void* keys, const void* values) = 0;
  virtual void FusedHalfUpdatePut(ep::Stream* stream, uint32_t n_keys, const void* keys,
                                  const void* values, const void* update, const float* lr,
                                  float scale) {
    UNIMPLEMENTED();
  }
  virtual bool IsFusionSupported() { return false; }
  virtual bool SnapshotExists(const std::string& name) = 0;
  virtual void LoadSnapshot(const std::string& name) = 0;
  virtual void LoadSnapshot(const std::string& name,
                            const std::function<void(KVIterator* iter)>& Hook) = 0;
  virtual void SaveSnapshot(const std::string& name) = 0;
};

}  // namespace embedding

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_KEY_VALUE_STORE_H_
