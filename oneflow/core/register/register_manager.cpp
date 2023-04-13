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
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/memory/chunk_manager.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

namespace {

struct PackedChunkInfo {
  MemoryCase mem_case;
  int64_t size;
  std::vector<const MemBlockProto*> blocks;
  PackedChunkInfo(const MemoryCase& mem) {
    mem_case = mem;
    size = 0;
  }
};

std::shared_ptr<ep::Device> GetDeviceByMemoryCase(const MemoryCase& mem_case) {
  return Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(mem_case.device_type(),
                                                                mem_case.device_id());
}

void InitDataRegst(Regst* regst, char* main_mem_ptr, char* separated_header_mem_ptr) {
  auto* rt_regst_desc = regst->regst_desc();
  size_t separated_header_mem_size = rt_regst_desc->SeparatedHeaderByteSize4OneRegst();
  char* cur_body_pointer = nullptr;
  char* cur_header_pointer = nullptr;
  if (separated_header_mem_size > 0) {
    MemoryCase host_mem_case = memory::MakeHostMemCase();
    if (separated_header_mem_ptr == nullptr) {
      separated_header_mem_ptr =
          Singleton<MemoryAllocator>::Get()->Allocate(host_mem_case, separated_header_mem_size);
    }
    cur_header_pointer = separated_header_mem_ptr;
    cur_body_pointer = main_mem_ptr;
  } else {
    CHECK(separated_header_mem_ptr == nullptr);
    cur_header_pointer = main_mem_ptr;
    if (main_mem_ptr == nullptr) {
      cur_body_pointer = nullptr;
    } else {
      cur_body_pointer =
          main_mem_ptr + rt_regst_desc->GetSoleBlobDesc()->AlignedByteSizeOfBlobHeader();
    }
  }
  if (regst->allocation_type() == RegstAllocationType::kStatic) {
    CHECK(cur_body_pointer != nullptr || rt_regst_desc->TotalBodyByteSize4AllRegst() == 0);
  } else if (regst->allocation_type() == RegstAllocationType::kStreamOrdered) {
    CHECK(cur_body_pointer == nullptr);
  } else {
    UNIMPLEMENTED();
  }
  regst->Init(cur_header_pointer);
  regst->ResetBodyMemPtr(cur_body_pointer);
}

}  // namespace

RegstMgr::RegstMgr() : stream_ordered_memory_allocation_enabled_(false) {
  stream_ordered_memory_allocation_enabled_ =
      ParseBooleanFromEnv("ONEFLOW_GRAPH_ENABLE_STREAM_ORDERED_MEMORY_ALLOCATION", false);
}

bool RegstMgr::IsStreamOrderedMemoryAllocationCase(const MemoryCase& mem_case) const {
  if (!stream_ordered_memory_allocation_enabled_) { return false; }
  const auto& device = GetDeviceByMemoryCase(mem_case);
  return device->IsStreamOrderedMemoryAllocationSupported();
}

void RegstMgr::AddPlan(
    const Plan& plan,
    const HashMap<std::string, vm::EagerBlobObject*>& variable_op_name2eager_blob_object) {
  int64_t this_machine_id = GlobalProcessCtx::Rank();

  HashMap<int64_t, char*> chunk_id2ptr;
  for (const ChunkProto& chunk : plan.block_chunk_list().chunk()) {
    if (chunk.machine_id() != this_machine_id) { continue; }
    if (chunk.mem_size() == 0) { continue; }
    if (IsStreamOrderedMemoryAllocationCase(chunk.mem_case())) { continue; }
    char* chunk_ptr = Singleton<ChunkMgr>::Get()->FindOrCreateChunk(chunk);
    CHECK(chunk_id2ptr.emplace(chunk.chunk_id(), chunk_ptr).second);
  }

  HashSet<int64_t> all_block_ids;
  HashMap<int64_t, PackedChunkInfo> zone_id2packed_chunk;
  for (const MemBlockProto& mem_block : plan.block_chunk_list().mem_block()) {
    if (mem_block.machine_id() != this_machine_id) { continue; }
    if (mem_block.mem_size() == 0) { continue; }
    const int64_t mem_block_id = mem_block.mem_block_id();
    CHECK(all_block_ids.insert(mem_block_id).second);

    if (mem_block.has_chunk_id()) {
      if (IsStreamOrderedMemoryAllocationCase(mem_block.mem_case())) {
        CHECK(mem_block.enable_reuse_mem());
        CHECK(stream_ordered_allocation_mem_block_ids_.emplace(mem_block_id).second);
        continue;
      }
      CHECK(mem_block.has_chunk_offset());
      CHECK(chunk_id2ptr.find(mem_block.chunk_id()) != chunk_id2ptr.end());
      char* mem_block_ptr = chunk_id2ptr.at(mem_block.chunk_id()) + mem_block.chunk_offset();
      CHECK(mem_block_id2ptr_.emplace(mem_block_id, mem_block_ptr).second);
      CHECK(!mem_block.has_variable_op_name());
    } else if (mem_block.has_variable_op_name()) {
      // NOTE(chengcheng): bind mem_block_ptr to variable blob header_ptr and body_ptr
      CHECK(!mem_block.enable_reuse_mem());
      const std::string& var_name = mem_block.variable_op_name();
      CHECK(!var_name.empty());
      auto it = variable_op_name2eager_blob_object.find(var_name);
      CHECK(it != variable_op_name2eager_blob_object.end())
          << " CANNOT find variable op name: " << var_name;
      CHECK(mem_block.has_is_separated_header());
      vm::EagerBlobObject* var_blob = it->second;
      CHECK(var_blob) << " variable op name: " << var_name << " in rank: " << this_machine_id
                      << " CANNNOT NULL.";
      if (mem_block.is_separated_header()) {
        CHECK_GE(var_blob->AlignedByteSizeOfBlobHeader(), mem_block.mem_size());
        CHECK_GE(mem_block.mem_size(), var_blob->ByteSizeOfBlobHeader());
        CHECK(mem_block_id2ptr_.emplace(mem_block_id, var_blob->mut_header_ptr()).second);
        CHECK(memory::IsHostMem(mem_block.mem_case()));
      } else {
        CHECK_GE(var_blob->AlignedByteSizeOfBlobBody(), mem_block.mem_size());
        CHECK_GE(mem_block.mem_size(), var_blob->ByteSizeOfBlobBody());
        CHECK(mem_block_id2ptr_.emplace(mem_block_id, var_blob->mut_dptr<char>()).second);
        // NOTE(chengcheng):
        //   CPU eager var tensor mem case is host_mem WITHOUT cuda pinned, but Lazy Complier
        //   will set variable op output blob mem_case with cuda pinned memory if this output
        //   blob has GPU op consume. We can JUST ignore this diff because it ONLY has little
        //   perf loss but correct.
        //   And this problem is NOT tensor.to("cuda") or tensor.to_global().
        CHECK(memory::EqualsIgnorePinnedDevice(mem_block.mem_case(), var_blob->mem_case()))
            << " variable op name: " << var_name << " in rank: " << this_machine_id
            << " bind eager tensor failed. The eager var tensor mem_case is : "
            << var_blob->mem_case().DebugString()
            << " but graph expected_mem block mem_case is : " << mem_block.mem_case().DebugString();
      }
    } else {
      int64_t zone_id = memory::GetMemCaseId(mem_block.mem_case());
      if (zone_id2packed_chunk.find(zone_id) == zone_id2packed_chunk.end()) {
        zone_id2packed_chunk.emplace(zone_id, PackedChunkInfo(mem_block.mem_case()));
      }
      PackedChunkInfo* packed_chunk = &(zone_id2packed_chunk.at(zone_id));
      packed_chunk->blocks.emplace_back(&mem_block);
      packed_chunk->size += mem_block.mem_size();
      CHECK(packed_chunk->mem_case == mem_block.mem_case());
    }
  }

  for (auto& pair : zone_id2packed_chunk) {
    PackedChunkInfo* packed_chunk = &pair.second;
    char* ptr =
        Singleton<MemoryAllocator>::Get()->Allocate(packed_chunk->mem_case, packed_chunk->size);
    // sort blocks as thrd id
    std::vector<const MemBlockProto*>* blocks = &(packed_chunk->blocks);
    std::sort(blocks->begin(), blocks->end(),
              [](const MemBlockProto* lhs, const MemBlockProto* rhs) {
                if (lhs->thrd_id_hint() == rhs->thrd_id_hint()) {
                  return lhs->mem_block_id() < rhs->mem_block_id();
                }
                return lhs->thrd_id_hint() < rhs->thrd_id_hint();
              });
    int64_t offset = 0;
    for (const MemBlockProto* block : packed_chunk->blocks) {
      CHECK(mem_block_id2ptr_.emplace(block->mem_block_id(), ptr + offset).second);
      offset += block->mem_size();
    }
    CHECK_EQ(offset, packed_chunk->size);
  }

  for (int64_t mem_block_id : all_block_ids) {
    if (mem_block_id2ptr_.find(mem_block_id) != mem_block_id2ptr_.end()) {
      CHECK(stream_ordered_allocation_mem_block_ids_.find(mem_block_id)
            == stream_ordered_allocation_mem_block_ids_.end());
    } else {
      CHECK(stream_ordered_allocation_mem_block_ids_.find(mem_block_id)
            != stream_ordered_allocation_mem_block_ids_.end());
    }
  }

  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != this_machine_id) { continue; }
    for (const auto& pair : task.produced_regst_desc()) {
      const RegstDescProto& regst_desc = pair.second;
      const int64_t regst_desc_id = regst_desc.regst_desc_id();
      CHECK(regst_desc_id2rt_regst_desc_
                .emplace(regst_desc_id, std::make_unique<const RtRegstDesc>(regst_desc))
                .second);
    }
  }
  for (const auto& pair : plan.ctrl_regst_desc_info().ctrl_regst_desc_id2producer_task_id()) {
    CHECK(ctrl_regst_desc_id2producer_task_id_.emplace(pair.first, pair.second).second);
  }
}

void RegstMgr::AddPlan(const Plan& plan) {
  HashMap<std::string, vm::EagerBlobObject*> variable_op_name2eager_blob_object;
  AddPlan(plan, variable_op_name2eager_blob_object);
}

void RegstMgr::NewRegsts(const RegstDescProto& regst_desc_proto,
                         std::function<void(Regst*)> OneRegstDone) {
  const int64_t regst_desc_id = regst_desc_proto.regst_desc_id();
  const RegstDescTypeProto& regst_desc_type = regst_desc_proto.regst_desc_type();
  const RtRegstDesc* rt_regst_desc = regst_desc_id2rt_regst_desc_.at(regst_desc_id).get();
  char* main_mem_ptr = nullptr;
  char* separated_header_mem_ptr = nullptr;
  int64_t mem_block_id = regst_desc_proto.mem_block_id();
  int64_t header_block_id = regst_desc_proto.separated_header_mem_block_id();
  if (mem_block_id != -1 && mem_block_id2ptr_.find(mem_block_id) != mem_block_id2ptr_.end()) {
    main_mem_ptr = mem_block_id2ptr_.at(mem_block_id) + regst_desc_proto.mem_block_offset();
  }
  if (header_block_id != -1 && mem_block_id2ptr_.find(header_block_id) != mem_block_id2ptr_.end()) {
    separated_header_mem_ptr = mem_block_id2ptr_.at(header_block_id);
  }
  RegstAllocationType allocation_type = stream_ordered_allocation_mem_block_ids_.find(mem_block_id)
                                                == stream_ordered_allocation_mem_block_ids_.end()
                                            ? RegstAllocationType::kStatic
                                            : RegstAllocationType::kStreamOrdered;
  for (int64_t i = 0; i < rt_regst_desc->register_num(); ++i) {
    Regst* regst = new Regst(rt_regst_desc, allocation_type);
    if (regst_desc_type.has_data_regst_desc()) {
      InitDataRegst(regst, main_mem_ptr, separated_header_mem_ptr);
      if (main_mem_ptr != nullptr) { main_mem_ptr += rt_regst_desc->MainByteSize4OneRegst(); }
      if (separated_header_mem_ptr != nullptr) {
        separated_header_mem_ptr += rt_regst_desc->SeparatedHeaderByteSize4OneRegst();
      }
    } else if (regst_desc_type.has_ctrl_regst_desc()) {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
    OneRegstDone(regst);
  }
}

const RtRegstDesc& RegstMgr::RegstDesc4RegstDescId(int64_t regst_desc_id) const {
  const auto& it = regst_desc_id2rt_regst_desc_.find(regst_desc_id);
  CHECK(it != regst_desc_id2rt_regst_desc_.end());
  return *it->second;
}

bool RegstMgr::HasRegstDescId(int64_t regst_desc_id) const {
  return regst_desc_id2rt_regst_desc_.find(regst_desc_id) != regst_desc_id2rt_regst_desc_.end();
}

int64_t RegstMgr::ProducerTaskId4RegstDescId(int64_t regst_desc_id) const {
  const auto& it = ctrl_regst_desc_id2producer_task_id_.find(regst_desc_id);
  CHECK(it != ctrl_regst_desc_id2producer_task_id_.end());
  return it->second;
}

bool RegstMgr::HasProducerTaskId4RegstDescId(int64_t regst_desc_id) const {
  return ctrl_regst_desc_id2producer_task_id_.find(regst_desc_id)
         != ctrl_regst_desc_id2producer_task_id_.end();
}

}  // namespace oneflow
