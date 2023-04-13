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
#ifndef ONEFLOW_CORE_EMBEDDING_PERSISTENT_TABLE_H_
#define ONEFLOW_CORE_EMBEDDING_PERSISTENT_TABLE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

namespace embedding {

struct PersistentTableOptions {
  std::string path;
  uint32_t key_size = 0;
  uint32_t value_size = 0;
  uint64_t target_chunk_size_mb = 4 * 1024;
  uint16_t physical_block_size = 4096;
  uint64_t capacity_hint = 0;
  bool read_only = false;
};

class PersistentTable {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentTable);
  PersistentTable() = default;
  virtual ~PersistentTable() = default;

  class Iterator {
   public:
    OF_DISALLOW_COPY_AND_MOVE(Iterator);
    Iterator() = default;
    virtual ~Iterator() = default;

    virtual void Next(uint32_t n_request, uint32_t* n_result, void* keys, void* values) = 0;
    virtual void Reset() = 0;
  };

  virtual uint32_t KeySize() const = 0;
  virtual uint32_t ValueSize() const = 0;
  virtual uint32_t LogicalBlockSize() const = 0;
  virtual void GetBlocks(uint32_t num_keys, const void* keys, void* blocks, uint32_t* offsets) = 0;
  virtual void Get(uint32_t num_keys, const void* keys, void* values, uint32_t* n_missing,
                   uint32_t* missing_indices) = 0;
  virtual void PutBlocks(uint32_t num_keys, const void* keys, const void* blocks) = 0;
  virtual void Put(uint32_t num_keys, const void* keys, const void* values) = 0;
  virtual bool SnapshotExists(const std::string& name) = 0;
  virtual void LoadSnapshot(const std::string& name) = 0;
  virtual void LoadSnapshot(const std::string& name,
                            const std::function<void(Iterator* iter)>& Hook) = 0;
  virtual void SaveSnapshot(const std::string& name) = 0;
  virtual Iterator* ReadSnapshot(const std::string& name) = 0;
};

std::unique_ptr<PersistentTable> NewPersistentTable(const PersistentTableOptions& options);

}  // namespace embedding

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_PERSISTENT_TABLE_H_
