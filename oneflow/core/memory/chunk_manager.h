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
#ifndef ONEFLOW_CORE_MEMORY_CHUNK_MANAGER_H_
#define ONEFLOW_CORE_MEMORY_CHUNK_MANAGER_H_

#include <mutex>

#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/memory/memory_block.pb.h"
#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

class ChunkMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChunkMgr);
  ChunkMgr() = default;
  ~ChunkMgr() = default;

  // Compiler
  void GetChunkProtosByMemZoneUniqueId(int64_t mem_zone_uid,
                                       std::vector<const ChunkProto*>* chunks) const;
  void AddChunkProto(const ChunkProto& chunk);

  // Runtime
  char* FindOrCreateChunk(const ChunkProto& chunk);

 private:
  // for master compiler in PlanUtil::GenMemBlockAndChunkWithVariableOpNames4Plan
  HashMap<int64_t, HashSet<int64_t>> mzuid2chunk_ids_;
  HashMap<int64_t, std::unique_ptr<ChunkProto>> chunk_id2chunk_proto_;

  struct ChunkWithPtr {
    char* ptr;
    ChunkProto chunk_proto;
    ChunkWithPtr(char* p, const ChunkProto& c_p) : ptr(p), chunk_proto(c_p) {}
  };

  // for runtime
  HashMap<int64_t, ChunkWithPtr> chunk_id2chunk_;
  std::mutex mutex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_CHUNK_MANAGER_H_
