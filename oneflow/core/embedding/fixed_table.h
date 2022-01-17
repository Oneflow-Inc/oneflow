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
#ifndef ONEFLOW_CORE_EMBEDDING_FIXED_TABLE_H_
#define ONEFLOW_CORE_EMBEDDING_FIXED_TABLE_H_

#include "oneflow/core/embedding/key_value_store.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace embedding {

struct FixedTableOptions {
  std::string path;
  uint32_t key_size = 0;
  uint32_t value_size = 0;
  uint64_t num_blocks_per_chunk = 4 * 1024 * 1024;
  uint16_t physical_block_size = 4096;
};

class FixedTable {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FixedTable);
  FixedTable() = default;
  virtual ~FixedTable() = default;

  class KeyIterator {
   public:
    OF_DISALLOW_COPY_AND_MOVE(KeyIterator);
    KeyIterator() = default;
    virtual ~KeyIterator() = default;

    virtual void Next(uint32_t num_keys, uint32_t* return_keys, void* keys) = 0;
  };

  virtual uint32_t KeySize() const = 0;
  virtual uint32_t ValueSize() const = 0;
  virtual uint32_t LogicalBlockSize() const = 0;
  virtual void GetBlocks(uint32_t num_keys, const void* keys, void* blocks, uint32_t* offsets) = 0;
  virtual void Get(uint32_t num_keys, const void* keys, void* values, uint32_t* n_missing,
                   uint32_t* missing_indices) = 0;
  virtual void PutBlocks(uint32_t num_keys, const void* keys, const void* blocks) = 0;
  virtual void Put(uint32_t num_keys, const void* keys, const void* values) = 0;
  virtual void WithKeyIterator(const std::function<void(KeyIterator* iter)>& fn) = 0;
  virtual bool SnapshotExists(const std::string& name) = 0;
  virtual void LoadSnapshot(const std::string& name) = 0;
  virtual void SaveSnapshot(const std::string& name) = 0;
};

std::unique_ptr<FixedTable> NewFixedTable(const FixedTableOptions& options);

}  // namespace embedding

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_FIXED_TABLE_H_
