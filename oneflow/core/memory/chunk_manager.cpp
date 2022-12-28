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
#include "oneflow/core/memory/chunk_manager.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {

void ChunkMgr::GetChunkProtosByMemZoneUniqueId(int64_t mem_zone_uid,
                                               std::vector<const ChunkProto*>* chunks) const {
  chunks->clear();
  auto chunk_ids_it = mzuid2chunk_ids_.find(mem_zone_uid);
  if (chunk_ids_it != mzuid2chunk_ids_.end()) {
    const auto& chunk_ids = chunk_ids_it->second;
    chunks->reserve(chunk_ids.size());
    for (int64_t chunk_id : chunk_ids) {
      auto chunk_it = chunk_id2chunk_proto_.find(chunk_id);
      CHECK(chunk_it != chunk_id2chunk_proto_.end());
      chunks->emplace_back(chunk_it->second.get());
    }
  }
}

void ChunkMgr::AddChunkProto(const ChunkProto& chunk) {
  const int64_t mem_zone_uid = memory::GetUniqueMemCaseId(chunk.machine_id(), chunk.mem_case());
  CHECK(
      chunk_id2chunk_proto_.emplace(chunk.chunk_id(), std::make_unique<ChunkProto>(chunk)).second);
  auto chunk_ids_it = mzuid2chunk_ids_.find(mem_zone_uid);
  if (chunk_ids_it == mzuid2chunk_ids_.end()) {
    chunk_ids_it = mzuid2chunk_ids_.emplace(mem_zone_uid, HashSet<int64_t>()).first;
  }
  CHECK(chunk_ids_it->second.insert(chunk.chunk_id()).second);
}

char* ChunkMgr::FindOrCreateChunk(const ChunkProto& chunk) {
  CHECK_EQ(GlobalProcessCtx::Rank(), chunk.machine_id());
  auto it = chunk_id2chunk_.find(chunk.chunk_id());
  if (it == chunk_id2chunk_.end()) {
    char* chunk_ptr =
        Singleton<MemoryAllocator>::Get()->Allocate(chunk.mem_case(), chunk.mem_size());
    it = chunk_id2chunk_.emplace(chunk.chunk_id(), ChunkWithPtr(chunk_ptr, chunk)).first;
  } else {
    const ChunkProto& store_proto = it->second.chunk_proto;
    CHECK_EQ(chunk.chunk_id(), store_proto.chunk_id());
    CHECK_EQ(chunk.machine_id(), store_proto.machine_id());
    CHECK(chunk.mem_case() == store_proto.mem_case());
    CHECK_EQ(chunk.mem_size(), store_proto.mem_size());
  }
  return it->second.ptr;
}

}  // namespace oneflow
