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
#include "oneflow/core/common/constant.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/graph/plan_task_graph.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/memory/chunk_manager.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

RegstDescProto* PlanUtil::GetSoleProducedDataRegst(TaskProto* task_proto) {
  RegstDescProto* ret = nullptr;
  for (auto& pair : *task_proto->mutable_produced_regst_desc()) {
    RegstDescProto* regst_desc = &pair.second;
    if (regst_desc->regst_desc_type().has_data_regst_desc()) {
      CHECK_ISNULL(ret);
      CHECK_EQ(regst_desc->regst_desc_type().data_regst_desc().lbi2blob_desc_size(), 1);
      ret = regst_desc;
    }
  }
  CHECK_NOTNULL(ret);
  return ret;
}

std::function<const TaskProto*(int64_t)> PlanUtil::MakeGetterTaskProto4TaskId(const Plan& plan) {
  auto task_id2task_proto = std::make_shared<HashMap<int64_t, const TaskProto*>>();
  for (const TaskProto& task_proto : plan.task()) {
    task_id2task_proto->emplace(task_proto.task_id(), &task_proto);
  }
  return [task_id2task_proto](int64_t task_id) { return task_id2task_proto->at(task_id); };
}

namespace {

void SetVariableOpNamesForVariableAndRepeatRegst(Plan* plan) {
  // NOTE(chengcheng): set variable_op_name before set separated header because var regst alway
  //  separated.
  HashMap<int64_t, std::string> regst_id2var_name;
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    if (task->exec_sequence().exec_node_size() == 1) {
      const auto& op_conf =
          PlanUtil::GetOpAttribute(plan, task->job_id(),
                                   task->exec_sequence().exec_node(0).kernel_conf())
              .op_conf();
      if (op_conf.has_variable_conf()) {
        RegstDescProto* regst = PlanUtil::GetSoleProducedDataRegst(task);
        regst_id2var_name.emplace(regst->regst_desc_id(), op_conf.name());
        regst->set_variable_op_name(op_conf.name());
      }
    }
  }

  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    if (task->task_type() == TaskType::kRepeat) {
      RegstDescProto* regst = PlanUtil::GetSoleProducedDataRegst(task);
      CHECK(regst->has_force_inplace_consumed_regst_desc_id());
      int64_t force_inplace_regst_id = regst->force_inplace_consumed_regst_desc_id();
      auto var_name_it = regst_id2var_name.find(force_inplace_regst_id);
      if (var_name_it != regst_id2var_name.end()) {
        regst->set_variable_op_name(var_name_it->second);
        VLOG(3) << " set var op name to repeat regst : " << regst->DebugString();
      }
    }
  }
}

}  // namespace

void PlanUtil::SetUniqueMemBlockId4UnreusedMemRegst(Plan* plan) {
  SetVariableOpNamesForVariableAndRepeatRegst(plan);

  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);

    for (auto& pair : *task->mutable_produced_regst_desc()) {
      RegstDescProto* regst_desc = &pair.second;
      if (regst_desc->mem_block_id() == -1) {
        CHECK_EQ(regst_desc->mem_block_offset(), -1);
        regst_desc->set_mem_block_id(Singleton<IDMgr>::Get()->NewMemBlockId());
        regst_desc->set_mem_block_offset(0);
      }

      RtRegstDesc rt_regst_desc(*regst_desc);
      int64_t regst_separated_size = rt_regst_desc.TotalSeparatedHeaderByteSize4AllRegst();
      if (regst_separated_size > 0) {
        int64_t separated_mem_block_id = Singleton<IDMgr>::Get()->NewMemBlockId();
        regst_desc->set_separated_header_mem_block_id(separated_mem_block_id);
      }
    }
  }
}

void PlanUtil::GenMemBlockAndChunk4Plan(Plan* plan) {
  HashSet<std::string> variable_op_names;
  PlanUtil::GenMemBlockAndChunkWithVariableOpNames4Plan(plan, variable_op_names);
}

namespace {

void GenChunkForMultiNNGraphMemoryReuseInMultiClient(
    Plan* plan, HashMap<int64_t, std::unique_ptr<MemBlockProto>>* mem_block_id2mem_block) {
  HashMap<int64_t, HashSet<MemBlockProto*>> mzuid2mem_blocks;

  for (auto& pair : *mem_block_id2mem_block) {
    MemBlockProto* mem_block = pair.second.get();
    CHECK(mem_block->has_chunk_id() == false);
    CHECK(mem_block->has_chunk_offset() == false);
    if (mem_block->has_variable_op_name()) { continue; }
    if (!mem_block->enable_reuse_mem()) { continue; }
    // NOTE(chengcheng):
    //   only reused mem in cuda device.
    //   special cpu memory like OFRecord pb and TensorBuffer CANNOT reused by another plan.
    if (memory::IsHostMem(mem_block->mem_case())) { continue; }
    int64_t mem_zone_uid =
        memory::GetUniqueMemCaseId(mem_block->machine_id(), mem_block->mem_case());
    auto it = mzuid2mem_blocks.find(mem_zone_uid);
    if (it == mzuid2mem_blocks.end()) {
      it = mzuid2mem_blocks.emplace(mem_zone_uid, HashSet<MemBlockProto*>()).first;
    }
    CHECK(it->second.insert(mem_block).second);
  }

  std::vector<ChunkProto> all_chunks;
  HashSet<int64_t> unique_chunk_ids;

  for (auto& pair : mzuid2mem_blocks) {
    int64_t mem_zone_uid = pair.first;
    std::vector<const ChunkProto*> exist_chunks;
    Singleton<ChunkMgr>::Get()->GetChunkProtosByMemZoneUniqueId(mem_zone_uid, &exist_chunks);
    auto chunk_it = exist_chunks.begin();
    auto& mem_blocks = pair.second;
    int64_t current_chunk_offset = 0;
    HashSet<MemBlockProto*> remain_blocks;
    for (auto mem_block_it = mem_blocks.begin(); mem_block_it != mem_blocks.end(); ++mem_block_it) {
      if (chunk_it == exist_chunks.end()) {
        // NOTE(chengcheng): it means that exist chunk has run out.
        CHECK(remain_blocks.insert(*mem_block_it).second);
      } else {
        // NOTE(chengcheng): find chunk which has enough space left.
        while (chunk_it != exist_chunks.end()
               && (current_chunk_offset + (*mem_block_it)->mem_size() > (*chunk_it)->mem_size())) {
          // NOTE(chengcheng): current chunk has no space left, so we move to next chunk.
          ++chunk_it;
          current_chunk_offset = 0;
        }
        if (chunk_it != exist_chunks.end()) {
          // NOTE(chengcheng): lucky, we find a appropriate chunk.
          MemBlockProto* mem_block = *mem_block_it;
          const ChunkProto* chunk = *chunk_it;
          CHECK_EQ(mem_block->machine_id(), chunk->machine_id());
          CHECK(mem_block->mem_case() == chunk->mem_case());
          CHECK_LE(current_chunk_offset + mem_block->mem_size(), chunk->mem_size());
          CHECK_GE(current_chunk_offset, 0);
          // CHECK_GT(mem_block->mem_size(), 0); NOTE(chengcheng): has mem block mem size = 0
          CHECK_GE(chunk->mem_size(), 0);
          mem_block->set_chunk_id(chunk->chunk_id());
          mem_block->set_chunk_offset(current_chunk_offset);
          current_chunk_offset += mem_block->mem_size();
          VLOG(3) << "Lazy nn.Graph Reused MemBlock :[" << mem_block->DebugString()
                  << "] to old Chunk :[" << chunk->DebugString() << "]\n";
        } else {
          // NOTE(chengcheng): sad, no chunk can used, so this mem block need to insert in remain.
          CHECK(remain_blocks.insert(*mem_block_it).second);
        }
      }
    }

    for (const ChunkProto* exist_chunk : exist_chunks) {
      all_chunks.emplace_back(*exist_chunk);
      CHECK(unique_chunk_ids.insert(exist_chunk->chunk_id()).second);
    }

    if (!remain_blocks.empty()) {
      auto remain_block_it = remain_blocks.begin();
      MemBlockProto* first_block = *remain_block_it;
      ChunkProto new_chunk;
      new_chunk.set_chunk_id(Singleton<IDMgr>::Get()->NewChunkId());
      new_chunk.set_machine_id(first_block->machine_id());
      *new_chunk.mutable_mem_case() = first_block->mem_case();
      new_chunk.set_mem_size(first_block->mem_size());
      first_block->set_chunk_id(new_chunk.chunk_id());
      first_block->set_chunk_offset(0);
      ++remain_block_it;
      VLOG(3) << "Lazy nn.Graph Add MemBlock :[" << first_block->DebugString() << "] to NewChunk :["
              << new_chunk.DebugString() << "]\n";

      while (remain_block_it != remain_blocks.end()) {
        MemBlockProto* this_block = *remain_block_it;
        CHECK_EQ(this_block->machine_id(), new_chunk.machine_id());
        CHECK(this_block->mem_case() == new_chunk.mem_case());
        this_block->set_chunk_id(new_chunk.chunk_id());
        this_block->set_chunk_offset(new_chunk.mem_size());
        new_chunk.set_mem_size(new_chunk.mem_size() + this_block->mem_size());
        VLOG(3) << "Lazy nn.Graph Add MemBlock :[" << this_block->DebugString()
                << "] to NewChunk :[" << new_chunk.DebugString() << "]\n";
        ++remain_block_it;
      }

      all_chunks.emplace_back(new_chunk);
      CHECK(unique_chunk_ids.insert(new_chunk.chunk_id()).second);

      Singleton<ChunkMgr>::Get()->AddChunkProto(new_chunk);
    }
  }

  CHECK_EQ(all_chunks.size(), unique_chunk_ids.size());

  for (const ChunkProto& chunk : all_chunks) {
    *(plan->mutable_block_chunk_list()->add_chunk()) = chunk;
  }
}

}  // namespace

void PlanUtil::MergeMemBlockIdByLogicalChainId(Plan* plan, const Job& job) {
  if (job.logical_chain_groups_size() == 0) { return; }
  HashMap<int64_t, HashMap<int64_t, int64_t>> logical_chain_id2machine_id2mem_block_id;

  for (int64_t i = 0; i < plan->task_size(); ++i) {
    TaskProto* task = plan->mutable_task(i);
    const StreamId stream_id = PlanUtil::GetStreamId(*task);
    int64_t machine_id = task->machine_id();
    DeviceType device_type = stream_id.device_id().device_type();
    // TODO(zwx): eliminate this special 'is cpu' determine
    if (device_type == DeviceType::kCPU) { continue; }
    if (!IsValidChainId(task->task_set_info().chain_id())) { continue; }
    int64_t logical_chain_id = task->task_set_info().chain_id();

    for (auto& pair : *(task->mutable_produced_regst_desc())) {
      RegstDescProto* regst_desc = &pair.second;
      if (regst_desc->mem_block_id() != -1 && regst_desc->enable_reuse_mem()
          && regst_desc->mem_case().device_type() == device_type
          && regst_desc->regst_desc_type().has_data_regst_desc()) {
        int64_t mem_block_id = regst_desc->mem_block_id();
        auto* rank2blocks = &(logical_chain_id2machine_id2mem_block_id[logical_chain_id]);
        if (rank2blocks->find(machine_id) == rank2blocks->end()) {
          rank2blocks->emplace(machine_id, mem_block_id);
        } else {
          CHECK_EQ(rank2blocks->at(machine_id), mem_block_id);
        }
      }
    }
  }

  HashMap<int64_t, int64_t> mem_block_id2merged_mem_block_id;
  for (const auto& logical_chain_group : job.logical_chain_groups()) {
    CHECK_GE(logical_chain_group.logical_chain_id_list_size(), 2);
    int64_t merged_logical_chain_id = logical_chain_group.logical_chain_id_list(0);
    CHECK(logical_chain_id2machine_id2mem_block_id.find(merged_logical_chain_id)
          != logical_chain_id2machine_id2mem_block_id.end());
    const auto& merged_rank2block =
        logical_chain_id2machine_id2mem_block_id.at(merged_logical_chain_id);
    for (int64_t i = 1; i < logical_chain_group.logical_chain_id_list_size(); ++i) {
      int64_t this_logical_chain_id = logical_chain_group.logical_chain_id_list(i);
      // NOTE(chengcheng): merge mem block id by each rank
      CHECK(logical_chain_id2machine_id2mem_block_id.find(this_logical_chain_id)
            != logical_chain_id2machine_id2mem_block_id.end());
      const auto& this_rank2block =
          logical_chain_id2machine_id2mem_block_id.at(this_logical_chain_id);
      for (const auto& pair : this_rank2block) {
        int64_t this_machine_id = pair.first;
        int64_t this_mem_block_id = pair.second;
        CHECK(merged_rank2block.find(this_machine_id) != merged_rank2block.end());
        int64_t merged_mem_block_id = merged_rank2block.at(this_machine_id);
        CHECK(mem_block_id2merged_mem_block_id.emplace(this_mem_block_id, merged_mem_block_id)
                  .second);
        VLOG(2) << " merge mem_block_id: " << this_mem_block_id << " to " << merged_mem_block_id;
      }
    }
  }

  for (int64_t i = 0; i < plan->task_size(); ++i) {
    TaskProto* task = plan->mutable_task(i);
    const StreamId stream_id = PlanUtil::GetStreamId(*task);
    DeviceType device_type = stream_id.device_id().device_type();
    // TODO(zwx): eliminate this special 'is cpu' determine
    if (device_type == DeviceType::kCPU) { continue; }
    if (!IsValidChainId(task->task_set_info().chain_id())) { continue; }

    for (auto& pair : *(task->mutable_produced_regst_desc())) {
      RegstDescProto* regst_desc = &pair.second;
      if (regst_desc->mem_block_id() != -1 && regst_desc->enable_reuse_mem()
          && regst_desc->mem_case().device_type() == device_type
          && regst_desc->regst_desc_type().has_data_regst_desc()) {
        int64_t mem_block_id = regst_desc->mem_block_id();
        if (mem_block_id2merged_mem_block_id.find(mem_block_id)
            != mem_block_id2merged_mem_block_id.end()) {
          // merge mem_block_id
          int64_t merged_mem_block_id = mem_block_id2merged_mem_block_id.at(mem_block_id);
          regst_desc->set_mem_block_id(merged_mem_block_id);
          const auto& data_regst = regst_desc->regst_desc_type().data_regst_desc();
          CHECK_GE(data_regst.lbi2blob_desc_size(), 1);
          const auto& lbi2blob_desc_pair = data_regst.lbi2blob_desc(0);
          std::string tensor_name = GenLogicalBlobName(lbi2blob_desc_pair.lbi());
          VLOG(3) << " regst: " << tensor_name << " merge mem block id " << mem_block_id << " to "
                  << merged_mem_block_id;
        }
      }
    }
  }
}

void PlanUtil::GenMemBlockAndChunkWithVariableOpNames4Plan(
    Plan* plan, const HashSet<std::string>& variable_op_names) {
  HashMap<int64_t, std::unique_ptr<MemBlockProto>> mem_block_id2mem_block;

  auto IsVariableRegst = [&](const TaskProto* task, std::string* name) -> bool {
    if (variable_op_names.empty()) { return false; }
    if (task->exec_sequence().exec_node_size() != 1) { return false; }
    const auto& op_conf =
        GetOpAttribute(plan, task->job_id(), task->exec_sequence().exec_node(0).kernel_conf())
            .op_conf();
    if (!op_conf.has_variable_conf()) { return false; }
    const std::string& var_name = op_conf.name();
    if (variable_op_names.find(var_name) == variable_op_names.end()) {
      LOG(WARNING) << " Oh no! Cannot find variable_op_name: " << var_name
                   << " in nn.Graph Compiler bind EagerTensor with VariableOp. "
                   << " \n But each variable need bind with eager tensor for init.";
      return false;
    }
    *name = var_name;
    return true;
  };

  auto GenMemBlock4RegstIfNeed = [&](RegstDescProto* regst_desc, const TaskProto* task) {
    const int64_t job_id = task->job_id();
    const int64_t machine_id = task->machine_id();
    const int64_t thrd_id = task->thrd_id();
    int64_t mem_block_id = regst_desc->mem_block_id();
    int64_t mem_block_offset = regst_desc->mem_block_offset();
    CHECK_NE(mem_block_id, -1);
    CHECK_NE(mem_block_offset, -1);

    std::string var_name;
    bool is_variable_regst = IsVariableRegst(task, &var_name);
    if (is_variable_regst) {
      CHECK(!var_name.empty());
      CHECK_EQ(regst_desc->register_num(), 1);
      CHECK_EQ(regst_desc->min_register_num(), 1);
      // NOTE(xuxiaoyu): this check cannot pass when open ZeRO
      // CHECK_EQ(regst_desc->max_register_num(), 1) << var_name;
      regst_desc->set_variable_op_name(var_name);
    }

    RtRegstDesc rt_regst_desc(*regst_desc);
    int64_t regst_main_size = rt_regst_desc.TotalMainByteSize4AllRegst();
    int64_t regst_separated_size = rt_regst_desc.TotalSeparatedHeaderByteSize4AllRegst();

    auto mem_block_it = mem_block_id2mem_block.find(mem_block_id);
    if (mem_block_it == mem_block_id2mem_block.end()) {
      MemBlockProto mem_block;
      mem_block.set_mem_block_id(mem_block_id);
      mem_block.add_job_id(job_id);
      mem_block.set_machine_id(machine_id);
      *(mem_block.mutable_mem_case()) = regst_desc->mem_case();
      mem_block.set_enable_reuse_mem(regst_desc->enable_reuse_mem());
      mem_block.set_mem_size(regst_main_size + mem_block_offset);
      mem_block.set_thrd_id_hint(thrd_id);
      if (is_variable_regst) {
        mem_block.set_variable_op_name(var_name);
        mem_block.set_is_separated_header(false);
      }
      CHECK(mem_block_id2mem_block
                .emplace(mem_block.mem_block_id(), std::make_unique<MemBlockProto>(mem_block))
                .second);
    } else {
      MemBlockProto* mem_block = mem_block_it->second.get();
      CHECK_EQ(mem_block->job_id(0), job_id);
      CHECK_EQ(mem_block->machine_id(), machine_id);
      CHECK(mem_block->mem_case() == regst_desc->mem_case());
      CHECK_EQ(mem_block->enable_reuse_mem(), regst_desc->enable_reuse_mem());
      if (mem_block->enable_reuse_mem()) {
        mem_block->set_mem_size(
            std::max(mem_block->mem_size(), regst_main_size + mem_block_offset));
      } else {
        CHECK_EQ(mem_block->mem_size(), regst_main_size);
        CHECK_EQ(mem_block_offset, 0);
      }
      if (is_variable_regst) {
        mem_block->set_variable_op_name(var_name);
        mem_block->set_is_separated_header(false);
      }
    }

    if (regst_separated_size > 0) {
      CHECK(regst_desc->has_separated_header_mem_block_id()) << regst_desc->DebugString();
      int64_t separated_mem_block_id = regst_desc->separated_header_mem_block_id();
      CHECK_NE(separated_mem_block_id, -1);
      if (mem_block_id2mem_block.find(separated_mem_block_id) == mem_block_id2mem_block.end()) {
        MemBlockProto mem_block;
        mem_block.set_mem_block_id(separated_mem_block_id);
        mem_block.add_job_id(job_id);
        mem_block.set_machine_id(machine_id);
        *(mem_block.mutable_mem_case()) = memory::GetPinnedHostMemoryCase(regst_desc->mem_case());
        mem_block.set_enable_reuse_mem(false);
        mem_block.set_mem_size(regst_separated_size);
        mem_block.set_thrd_id_hint(thrd_id);
        if (is_variable_regst) {
          mem_block.set_variable_op_name(var_name);
          mem_block.set_is_separated_header(true);
        }
        CHECK(mem_block_id2mem_block
                  .emplace(mem_block.mem_block_id(), std::make_unique<MemBlockProto>(mem_block))
                  .second);
      } else {
        MemBlockProto* mem_block = mem_block_id2mem_block.at(separated_mem_block_id).get();
        CHECK_EQ(mem_block->job_id(0), job_id);
        CHECK_EQ(mem_block->machine_id(), machine_id);
        CHECK(mem_block->mem_case() == memory::GetPinnedHostMemoryCase(regst_desc->mem_case()));
        CHECK_EQ(mem_block->enable_reuse_mem(), false);
        CHECK_EQ(mem_block->mem_size(), regst_separated_size);
        if (is_variable_regst) {
          mem_block->set_variable_op_name(var_name);
          mem_block->set_is_separated_header(true);
        }
      }
    }
  };

  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      GenMemBlock4RegstIfNeed(&pair.second, task);
    }
  }

  GenChunkForMultiNNGraphMemoryReuseInMultiClient(plan, &mem_block_id2mem_block);

  for (const auto& pair : mem_block_id2mem_block) {
    *(plan->mutable_block_chunk_list()->add_mem_block()) = *(pair.second);
  }
}

void PlanUtil::CleanUselessMemBlockAndCheckValid(Plan* plan) {
  HashMap<int64_t, ChunkProto> chunk_id2chunk;
  HashMap<int64_t, MemBlockProto> mem_block_id2mem_block;
  for (const auto& chunk : plan->block_chunk_list().chunk()) {
    CHECK(chunk_id2chunk.emplace(chunk.chunk_id(), chunk).second);
  }
  for (const auto& mem_block : plan->block_chunk_list().mem_block()) {
    CHECK(mem_block_id2mem_block.emplace(mem_block.mem_block_id(), mem_block).second);
  }
  plan->mutable_block_chunk_list()->clear_mem_block();

  HashMap<int64_t, HashSet<int64_t>> chunk_id2job_ids;
  HashMap<int64_t, HashSet<int64_t>> mem_block_id2job_ids;
  for (const auto& pair : chunk_id2chunk) {
    for (int64_t job_id : pair.second.job_id()) {
      CHECK(chunk_id2job_ids[pair.first].insert(job_id).second);
    }
  }
  for (const auto& pair : mem_block_id2mem_block) {
    for (int64_t job_id : pair.second.job_id()) {
      CHECK(mem_block_id2job_ids[pair.first].insert(job_id).second);
    }
  }

  HashSet<int64_t> valid_mem_block_ids;
  for (const TaskProto& task : plan->task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      const RegstDescProto& regst = pair.second;
      RtRegstDesc rt_regst(regst);
      int64_t regst_size = rt_regst.TotalMainByteSize4AllRegst();
      CHECK(mem_block_id2mem_block.find(regst.mem_block_id()) != mem_block_id2mem_block.end());
      const MemBlockProto& mem_block = mem_block_id2mem_block.at(regst.mem_block_id());
      CHECK_GE(mem_block.mem_size(), regst.mem_block_offset() + regst_size);
      CHECK_EQ(task.machine_id(), mem_block.machine_id());
      CHECK_EQ(mem_block.enable_reuse_mem(), regst.enable_reuse_mem());
      CHECK(mem_block.mem_case() == regst.mem_case());
      const auto& job_ids = mem_block_id2job_ids[regst.mem_block_id()];
      CHECK(job_ids.find(task.job_id()) != job_ids.end());
      valid_mem_block_ids.insert(regst.mem_block_id());

      // separated_header
      int64_t separated_header_mem_size = rt_regst.TotalSeparatedHeaderByteSize4AllRegst();
      if (separated_header_mem_size > 0) {
        int64_t header_block_id = regst.separated_header_mem_block_id();
        CHECK_NE(header_block_id, -1);
        CHECK(mem_block_id2mem_block.find(header_block_id) != mem_block_id2mem_block.end());
        const MemBlockProto& header_mem_block = mem_block_id2mem_block.at(header_block_id);
        CHECK_EQ(header_mem_block.mem_size(), separated_header_mem_size);
        CHECK_EQ(task.machine_id(), header_mem_block.machine_id());
        CHECK(header_mem_block.mem_case() == memory::GetPinnedHostMemoryCase(regst.mem_case()));
        CHECK(header_mem_block.enable_reuse_mem() == false);
        const auto& header_block_job_ids = mem_block_id2job_ids[header_block_id];
        CHECK(header_block_job_ids.find(task.job_id()) != header_block_job_ids.end());
        valid_mem_block_ids.insert(regst.separated_header_mem_block_id());
      }
    }
  }

  HashSet<int64_t> useless_mem_block_ids;
  HashSet<int64_t> valid_chunk_ids;
  for (const auto& pair : mem_block_id2mem_block) {
    if (valid_mem_block_ids.find(pair.first) == valid_mem_block_ids.end()) {
      CHECK(useless_mem_block_ids.insert(pair.first).second);
      continue;
    }
    const MemBlockProto& mem_block = pair.second;
    if (mem_block.has_chunk_id()) {
      CHECK(mem_block.has_chunk_offset());
      CHECK(mem_block.enable_reuse_mem());
      CHECK(chunk_id2chunk.find(mem_block.chunk_id()) != chunk_id2chunk.end());
      const ChunkProto& chunk = chunk_id2chunk.at(mem_block.chunk_id());
      CHECK_GE(chunk.mem_size(), mem_block.chunk_offset() + mem_block.mem_size());
      CHECK_EQ(mem_block.job_id_size(), 1);
      CHECK_GE(chunk.job_id_size(), 1);
      const HashSet<int64_t>& chunk_job_ids = chunk_id2job_ids.at(chunk.chunk_id());
      CHECK(chunk_job_ids.find(mem_block.job_id(0)) != chunk_job_ids.end());
      valid_chunk_ids.insert(mem_block.chunk_id());
    }
  }
  CHECK_EQ(valid_chunk_ids.size(), chunk_id2chunk.size());

  for (int64_t useless_block_id : useless_mem_block_ids) {
    mem_block_id2mem_block.erase(useless_block_id);
  }

  for (const auto& pair : mem_block_id2mem_block) {
    *(plan->mutable_block_chunk_list()->add_mem_block()) = pair.second;
  }
}

void PlanUtil::ToDotFile(const Plan& plan, const std::string& filepath) {
  const auto& process_ranks = Singleton<ResourceDesc, ForSession>::Get()->process_ranks();
  size_t gpu_device_num =
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDeviceCount(DeviceType::kCUDA);
  std::map<int64_t, std::map<int64_t, std::vector<std::vector<std::string>>>>
      machine_id2job_id_device_id2node_list;
  for (size_t i : process_ranks) {
    for (const auto& pair : plan.job_confs().job_id2job_conf()) {
      machine_id2job_id_device_id2node_list[i][pair.first].resize(gpu_device_num);
    }
  }
  std::map<int64_t, std::map<int64_t, std::vector<std::string>>> machine_id2job_id2host_node_list;
  std::vector<std::string> main_node_list;
  std::vector<std::string> copy_comm_net_node_list;
  HashSet<int64_t> ctrl_regst_desc_ids;
  HashMap<int64_t, HashMap<int64_t, std::string>> task_id2consumer_regst_id2name;
  HashMap<int64_t, std::string> task_id2op_name;
  HashMap<int64_t, std::vector<int64_t>> task_id2producer_task_ids;
  std::vector<std::set<int64_t>> machine_id2device_id2node_list_job_ids(process_ranks.size());
  std::vector<std::set<int64_t>> machine_id2host_node_list_job_ids(process_ranks.size());

  auto InsertNodeDefByTaskProto = [&](const TaskProto& task_proto, const std::string& node_def,
                                      const std::string& pass_tag) {
    if (task_proto.task_type() == TaskType::kCopyCommNet) {
      copy_comm_net_node_list.emplace_back(node_def);
      return;
    }
    if (pass_tag == kNoPassTag) {
      const StreamId stream_id = PlanUtil::GetStreamId(task_proto);
      if (stream_id.device_id().device_type() == DeviceType::kCUDA) {
        machine_id2job_id_device_id2node_list[task_proto.machine_id()][task_proto.job_id()]
                                             [stream_id.device_id().device_index()]
                                                 .emplace_back(node_def);
        machine_id2device_id2node_list_job_ids[task_proto.machine_id()].insert(task_proto.job_id());
      } else {
        machine_id2job_id2host_node_list[task_proto.machine_id()][task_proto.job_id()].emplace_back(
            node_def);
        machine_id2host_node_list_job_ids[task_proto.machine_id()].insert(task_proto.job_id());
      }
    } else if (pass_tag == kMainOp) {
      main_node_list.emplace_back(node_def);
    } else {
      UNIMPLEMENTED();
    }
  };

  auto GenEdgeColorStr = [](const RegstDescTypeProto& type) {
    if (type.has_ctrl_regst_desc()) { return "fontcolor=\"gray65\",color=\"gray65\""; }
    return "fontcolor=\"gray15\",color=\"gray15\"";
  };

  auto IsEsac2ReentrantLockEdge = [](const std::string& src_name, const std::string& dst_name) {
    if (src_name.find("Esac") != std::string::npos
        && dst_name.find("ReentrantLock") != std::string::npos) {
      return true;
    }
    return false;
  };

  auto IsEsacNode = [](const std::string& name) {
    if (name.find("Esac") != std::string::npos) { return true; }
    return false;
  };

  auto log_stream = TeePersistentLogStream::Create(filepath);
  // task node
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      const RegstDescProto& regst = pair.second;
      for (int64_t consumer_task_id : regst.consumer_task_id()) {
        task_id2producer_task_ids[consumer_task_id].emplace_back(task_proto.task_id());
      }
    }
  }

  for (const TaskProto& task_proto : plan.task()) {
    std::string task_id_str = "task" + std::to_string(task_proto.task_id());
    std::string task_class = task_id_str;
    for (const auto& in_task_id : task_id2producer_task_ids[task_proto.task_id()]) {
      task_class += " in" + std::to_string(in_task_id);
    }
    for (const auto& pair : task_proto.produced_regst_desc()) {
      const RegstDescProto& regst = pair.second;
      for (int64_t consumer_task_id : regst.consumer_task_id()) {
        task_class += " out" + std::to_string(consumer_task_id);
      }
    }
    task_class += " job_id" + std::to_string(task_proto.job_id());
    task_class += " machine_id" + std::to_string(task_proto.machine_id());
    std::string node_def = task_id_str + "[class=\"" + task_class + "\",label=\"{{";
    node_def += std::to_string(task_proto.task_id()) + ":" + std::to_string(task_proto.machine_id())
                + "\\n";
    std::string op_name = "";
    std::string pass_tag = kNoPassTag;
    for (const ExecNodeProto& exec_node : task_proto.exec_sequence().exec_node()) {
      const auto& op_conf =
          GetOpAttribute(&plan, task_proto.job_id(), exec_node.kernel_conf()).op_conf();
      op_name += op_conf.name();
      if (op_conf.has_pass_tag()) { pass_tag = op_conf.pass_tag(); }
    }
    task_id2op_name[task_proto.task_id()] = op_name;
    node_def += op_name;
    size_t index = 0;
    for (const auto& pair : task_proto.produced_regst_desc()) {
      std::string regst_id = std::to_string(pair.second.regst_desc_id());
      if (index % 2 == 0) {
        node_def += "}|{";
      } else {
        node_def += "|";
      }
      // node_def += "<regst_desc_" + regst_id + ">";
      node_def += (pair.first + ":" + regst_id + ":" + std::to_string(pair.second.register_num()));
      ++index;
    }
    node_def += "}}";
    node_def +=
        ("\",tooltip=\"" + TaskType_Name(task_proto.task_type()) + "  "
         + std::to_string(task_proto.task_id()) + "-" + std::to_string(task_proto.machine_id())
         + ":" + std::to_string(task_proto.thrd_id()) + ":"
         + std::to_string(task_proto.parallel_ctx().parallel_id())
         + "\", shape=record, style=\"rounded,filled\""
         + ",colorscheme=set312, fillcolor=" + std::to_string((task_proto.job_id() % 12) + 1));
    if (IsEsacNode(op_name)) { node_def += ",width=5,height=1.5"; }
    node_def += "];\n";
    InsertNodeDefByTaskProto(task_proto, node_def, pass_tag);
    for (const auto& pair : task_proto.consumed_regst_desc_id()) {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        task_id2consumer_regst_id2name[task_proto.task_id()][regst_desc_id] = pair.first;
      }
    }
  }

  log_stream << "digraph merged_plan_graph {\n";
  log_stream << "#splines=\"ortho\";\n";
  log_stream << "#rankdir=TB;\n";
  log_stream << "#nodesep=1.3;\n";
  log_stream << "#ranksep=1.3;\n";
  log_stream << "node[color=\"gray\"];\n";
  // main_node and copy_comm_net_node graph
  for (const std::string& main_node : main_node_list) { log_stream << main_node; }
  for (const std::string& copy_comm_net_node : copy_comm_net_node_list) {
    log_stream << copy_comm_net_node;
  }
  // sub graph
  for (size_t machine_id : process_ranks) {
    std::string machine_name = "machine_" + std::to_string(machine_id);
    log_stream << "subgraph cluster_" << machine_name << " { label = \"" << machine_name << "\";\n";
    log_stream << "style=\"rounded\";\n";
    {
      for (const auto& job_id : machine_id2host_node_list_job_ids[machine_id]) {
        std::string job_name = plan.job_confs().job_id2job_conf().at(job_id).job_name();
        job_name += (std::string(":") + std::to_string(job_id));
        if (job_id != plan.job_confs().job_id2job_conf().size() - 1) {
          log_stream << "subgraph cluster_job_" << std::to_string(job_id) << " { label = \""
                     << job_name << "\";\n";
          log_stream << "style=\"rounded\";\n";
        }
        for (const std::string& host_node_def :
             machine_id2job_id2host_node_list[machine_id][job_id]) {
          log_stream << host_node_def;
        }
        if (machine_id2device_id2node_list_job_ids[machine_id].find(job_id)
            != machine_id2device_id2node_list_job_ids[machine_id].end()) {
          for (size_t device_id = 0; device_id < gpu_device_num; ++device_id) {
            std::string device_name = machine_name + "_device_" + std::to_string(device_id);
            log_stream << "#subgraph cluster_" << device_name << " { label = \"" << device_name
                       << "\";\n";
            log_stream << "#color=\"skyblue\";\n";
            log_stream << "#fillcolor=\"azure\";\n";
            log_stream << "#style=\"rounded,filled\";\n";
            for (const auto& device_node_def :
                 machine_id2job_id_device_id2node_list[machine_id][job_id][device_id]) {
              log_stream << device_node_def;
            }
            log_stream << "#}\n";
          }
          machine_id2device_id2node_list_job_ids[machine_id].erase(job_id);
        }

        if (job_id != plan.job_confs().job_id2job_conf().size() - 1) { log_stream << "}\n"; }
      }
      for (const auto& job_id : machine_id2device_id2node_list_job_ids[machine_id]) {
        std::string job_name = plan.job_confs().job_id2job_conf().at(job_id).job_name();
        job_name += (std::string(":") + std::to_string(job_id));
        if (job_id != plan.job_confs().job_id2job_conf().size() - 1) {
          log_stream << "subgraph cluster_job_" << std::to_string(job_id) << " { label = \""
                     << job_name << "\";\n";
          log_stream << "style=\"rounded\";\n";
        }
        for (size_t device_id = 0; device_id < gpu_device_num; ++device_id) {
          std::string device_name = machine_name + "_device_" + std::to_string(device_id);
          log_stream << "#subgraph cluster_" << device_name << " { label = \"" << device_name
                     << "\";\n";
          log_stream << "#color=\"skyblue\";\n";
          log_stream << "#fillcolor=\"azure\";\n";
          log_stream << "#style=\"rounded,filled\";\n";
          for (const auto& device_node_def :
               machine_id2job_id_device_id2node_list[machine_id][job_id][device_id]) {
            log_stream << device_node_def;
          }
          log_stream << "#}\n";
        }
        if (job_id != plan.job_confs().job_id2job_conf().size() - 1) { log_stream << "}\n"; }
      }
    }
    log_stream << "}\n";
  }

  // produce/consume edge
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      const RegstDescProto& regst = pair.second;
      std::string src_node = "task" + std::to_string(task_proto.task_id());
      // src_node += ":regst_desc_" + std::to_string(regst.regst_desc_id());
      for (int64_t consumer_task_id : regst.consumer_task_id()) {
        std::string dst_node = "task" + std::to_string(consumer_task_id);
        // dst_node +=  ":task_node_" + std::to_string(consumer_task_id);
        std::string consumer_regst_name =
            task_id2consumer_regst_id2name[consumer_task_id][regst.regst_desc_id()];
        std::string consumer_op_name = task_id2op_name[consumer_task_id];
        std::string producer_regst_name = pair.first;
        std::string producer_op_name = task_id2op_name[task_proto.task_id()];
        std::string tooltip = producer_op_name + " : " + producer_regst_name + " -> "
                              + consumer_op_name + " : " + consumer_regst_name;
        if (IsEsac2ReentrantLockEdge(producer_op_name, consumer_op_name)) {
          log_stream << dst_node << "->" << src_node
                     << "[arrowhead=\"invempty\",fontcolor=\"red\",color=\"red\",taillabel=\""
                     << consumer_regst_name << "\",tailtooltip=\"" << tooltip;
        } else {
          log_stream << src_node << "->" << dst_node << "["
                     << GenEdgeColorStr(regst.regst_desc_type()) << ",headlabel=\""
                     << consumer_regst_name << "\",headtooltip=\"" << tooltip;
        }
        log_stream << "\",tooltip=\"" << tooltip << "\",arrowsize=0.5,labeldistance=1.5,penwidth=2"
                   << "];\n";
      }
    }
  }
  log_stream << "}\n";
}

std::function<RegstDescProto*(int64_t)> PlanUtil::MakeMutRegstDesc4Id(Plan* plan) {
  auto regst_desc_id2regst_desc = std::make_shared<HashMap<int64_t, RegstDescProto*>>();
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      int64_t regst_desc_id = pair.second.regst_desc_id();
      regst_desc_id2regst_desc->insert({regst_desc_id, &pair.second});
    }
  }
  return [regst_desc_id2regst_desc](int64_t regst_desc_id) -> RegstDescProto* {
    return regst_desc_id2regst_desc->at(regst_desc_id);
  };
}

void PlanUtil::SetForceInplaceMemBlock(Plan* plan) {
  auto RegstDesc4Id = MakeMutRegstDesc4Id(plan);
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      RegstDescProto* regst_desc = &pair.second;
      if (regst_desc->has_force_inplace_consumed_regst_desc_id()) {
        int64_t force_id = regst_desc->force_inplace_consumed_regst_desc_id();
        const RegstDescProto* in_regst_desc = RegstDesc4Id(force_id);
        CHECK(!in_regst_desc->enable_reuse_mem());
        CHECK(!regst_desc->enable_reuse_mem());
        CHECK_NE(in_regst_desc->mem_block_id(), -1);
        CHECK_EQ(in_regst_desc->mem_block_offset(), 0);
        CHECK_EQ(regst_desc->mem_block_offset(), 0);
        CHECK_EQ(in_regst_desc->register_num(), regst_desc->register_num());
        CHECK(in_regst_desc->mem_case() == regst_desc->mem_case());
        RtRegstDesc in_regst_rt(*in_regst_desc);
        RtRegstDesc regst_rt(*regst_desc);
        CHECK_EQ(in_regst_rt.TotalByteSize4AllRegst(), regst_rt.TotalByteSize4AllRegst());
        CHECK_EQ(in_regst_rt.TotalMainByteSize4AllRegst(), regst_rt.TotalMainByteSize4AllRegst());
        CHECK_EQ(in_regst_rt.TotalSeparatedHeaderByteSize4AllRegst(),
                 regst_rt.TotalSeparatedHeaderByteSize4AllRegst());
        regst_desc->set_mem_block_id(in_regst_desc->mem_block_id());
        regst_desc->set_inplace_consumed_regst_desc_id(force_id);
        if (in_regst_desc->has_separated_header_mem_block_id()) {
          CHECK(regst_desc->has_separated_header_mem_block_id());
          regst_desc->set_separated_header_mem_block_id(
              in_regst_desc->separated_header_mem_block_id());
        }
        VLOG(3) << " set force inplace from " << regst_desc->DebugString() << " to "
                << in_regst_desc->DebugString();
      }
    }
  }
}

void PlanUtil::DumpCtrlRegstInfoToPlan(Plan* plan) {
  auto* ctrl_regst_desc_id2producer_task_id =
      plan->mutable_ctrl_regst_desc_info()->mutable_ctrl_regst_desc_id2producer_task_id();
  for (const TaskProto& task : plan->task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      if (pair.second.regst_desc_type().has_ctrl_regst_desc()) {
        ctrl_regst_desc_id2producer_task_id->insert(
            {pair.second.regst_desc_id(), pair.second.producer_task_id()});
      }
    }
  }
}

namespace {

bool IsCollectiveBoxingTaskType(TaskType task_type) {
  return task_type == TaskType::kCollectiveBoxingGeneric;
}

bool IsCollectiveBoxingNode(const PlanTaskNode* node) {
  const TaskType task_type = node->task_proto()->task_type();
  return task_type == TaskType::kCollectiveBoxingGeneric;
}

const boxing::collective::RankDesc& GetRankDesc(const OperatorConf& conf) {
  if (conf.has_collective_boxing_generic_conf()) {
    return conf.collective_boxing_generic_conf().rank_desc();
  } else {
    UNIMPLEMENTED();
  }
}

const boxing::collective::RankDesc& GetRankDesc(Plan* plan, const TaskProto& task_proto) {
  CHECK_EQ(task_proto.exec_sequence().exec_node_size(), 1);
  return GetRankDesc(PlanUtil::GetOpAttribute(plan, task_proto.job_id(),
                                              task_proto.exec_sequence().exec_node(0).kernel_conf())
                         .op_conf());
}

struct CollectiveBoxingRequestInfo {
  boxing::collective::OpDesc op_desc;
  std::map<int64_t, const PlanTaskNode*> rank2node;
  int64_t order;
  int64_t dependency_depth;
};

void GetDeviceDesc(const TaskProto* task_proto, boxing::collective::DeviceDesc* device_desc) {
  device_desc->set_machine_id(task_proto->machine_id());
  const StreamId stream_id = PlanUtil::GetStreamId(*task_proto);
  const DeviceId& device_id = stream_id.device_id();
  device_desc->set_device_type(device_id.device_type());
  device_desc->set_device_id(device_id.device_index());
}

}  // namespace

void PlanUtil::GenCollectiveBoxingPlan(Job* job, Plan* plan) {
  using namespace boxing::collective;

  RequestSet* request_set = &(*plan->mutable_collective_boxing_plan()
                                   ->mutable_job_id2request_set())[GlobalJobDesc().job_id()];
  const int64_t cb_task_count = std::count_if(
      plan->task().cbegin(), plan->task().cend(),
      [](const TaskProto& task) { return IsCollectiveBoxingTaskType(task.task_type()); });
  if (cb_task_count == 0) { return; }

  PlanTaskGraph plan_task_graph(*plan);
  int64_t dependency_depth = 0;
  int64_t order = 0;
  HashSet<const PlanTaskNode*> all_visited;
  while (true) {
    std::list<const PlanTaskNode*> src_nodes;
    plan_task_graph.ForEachNode([&](const PlanTaskNode* node) {
      if (all_visited.count(node) != 0) { return; }
      int64_t in_cnt = 0;
      node->ForEachNodeOnInEdge([&](const PlanTaskNode* node_on_in_edge) {
        if (all_visited.count(node_on_in_edge) != 0) { return; }
        in_cnt += 1;
      });
      if (in_cnt == 0) { src_nodes.emplace_back(node); }
    });
    if (src_nodes.empty()) { break; }
    auto ForEachNodeOnInEdge = [&](const PlanTaskNode* node,
                                   const std::function<void(const PlanTaskNode*)>& Handler) {
      node->ForEachNodeOnInEdge([&](const PlanTaskNode* node_on_in_edge) {
        if (all_visited.count(node_on_in_edge) == 0) { Handler(node_on_in_edge); }
      });
    };
    auto ForEachNodeOnOutEdge = [&](const PlanTaskNode* node,
                                    const std::function<void(const PlanTaskNode*)>& Handler) {
      if (!IsCollectiveBoxingNode(node)) {
        node->ForEachNodeOnOutEdge([&](const PlanTaskNode* node_on_out_edge) {
          bool has_unvisited_collective_boxing_node_on_in_edges = false;
          node_on_out_edge->ForEachNodeOnInEdge([&](const PlanTaskNode* node_on_in_edge) {
            if (!has_unvisited_collective_boxing_node_on_in_edges
                && IsCollectiveBoxingNode(node_on_in_edge)
                && all_visited.count(node_on_in_edge) == 0) {
              has_unvisited_collective_boxing_node_on_in_edges = true;
            }
          });
          if (!has_unvisited_collective_boxing_node_on_in_edges) { Handler(node_on_out_edge); }
        });
      }
    };
    HashSet<const PlanTaskNode*> visited;
    std::vector<const PlanTaskNode*> collective_boxing_nodes;
    plan_task_graph.TopoForEachNode(src_nodes, ForEachNodeOnInEdge, ForEachNodeOnOutEdge,
                                    [&](const PlanTaskNode* node) {
                                      visited.insert(node);
                                      if (IsCollectiveBoxingNode(node)) {
                                        collective_boxing_nodes.emplace_back(node);
                                      }
                                    });
    if (collective_boxing_nodes.empty()) { break; }
    HashMap<std::string, CollectiveBoxingRequestInfo> name2request_info;
    for (const PlanTaskNode* node : collective_boxing_nodes) {
      const TaskProto* task_proto = node->task_proto();
      const RankDesc& rank_desc = GetRankDesc(plan, *task_proto);
      CHECK_GE(rank_desc.rank(), 0);
      CHECK_LT(rank_desc.rank(), rank_desc.op_desc().num_ranks());
      const std::string& name = rank_desc.op_desc().name();
      boxing::collective::DeviceDesc device_desc;
      GetDeviceDesc(task_proto, &device_desc);
      auto it = name2request_info.find(name);
      if (it == name2request_info.end()) {
        CollectiveBoxingRequestInfo request_info{
            .op_desc = rank_desc.op_desc(),
            .rank2node = {std::make_pair(rank_desc.rank(), node)},
            .order = order,
            .dependency_depth = dependency_depth,
        };
        name2request_info.emplace(std::make_pair(name, std::move(request_info)));
        order += 1;
      } else {
        CHECK(it->second.op_desc == rank_desc.op_desc());
        CHECK(it->second.rank2node.emplace(std::make_pair(rank_desc.rank(), node)).second);
      }
    }
    int64_t collected = 0;
    for (const auto& name7request_info : name2request_info) {
      const CollectiveBoxingRequestInfo& info = name7request_info.second;
      if (info.rank2node.size() == info.op_desc.num_ranks()) {
        collected += 1;
        boxing::collective::RequestDesc* request_desc = request_set->mutable_request()->Add();
        *request_desc->mutable_op_desc() = info.op_desc;
        for (int64_t i = 0; i < info.op_desc.num_ranks(); ++i) {
          GetDeviceDesc(info.rank2node.at(i)->task_proto(),
                        request_desc->mutable_device_set()->mutable_device()->Add());
        }
        request_desc->set_order(info.order);
        request_desc->set_dependency_depth(info.dependency_depth);
      } else {
        CHECK_LT(info.rank2node.size(), info.op_desc.num_ranks());
        for (const auto& pair : info.rank2node) { visited.erase(pair.second); }
      }
    }
    CHECK_GT(collected, 0);
    all_visited.insert(visited.begin(), visited.end());
    ++dependency_depth;
  }
}

void PlanUtil::GenRegisterHint(Plan* plan) {
  HashSet<int64_t> multi_regst_regst_desc_ids;
  for (const TaskProto& task : plan->task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      if (pair.second.register_num() != 1 || task.task_type() == TaskType::kRepeat) {
        multi_regst_regst_desc_ids.emplace(pair.second.regst_desc_id());
      }
    }
  }
  for (TaskProto& task : *(plan->mutable_task())) {
    bool all_register_num_eq_one = true;
    for (const auto& pair : task.produced_regst_desc()) {
      if (pair.second.register_num() != 1) {
        all_register_num_eq_one = false;
        break;
      }
    }
    for (const auto& pair : task.consumed_regst_desc_id()) {
      if (!all_register_num_eq_one) { break; }
      for (auto regst_desc_id : pair.second.regst_desc_id()) {
        if (multi_regst_regst_desc_ids.count(regst_desc_id) > 0) {
          all_register_num_eq_one = false;
          break;
        }
      }
    }
    task.set_all_register_num_eq_one_hint(all_register_num_eq_one);
  }
}

namespace {

struct MemBlockMemoryInfo {
  int64_t mem_block_id;
  int64_t mem_block_mem_size;
  int64_t regst_num;
  std::vector<int64_t> ordered_regst_desc_id;
  MemBlockMemoryInfo() : mem_block_id(-1), mem_block_mem_size(-1), regst_num(-1) {}
};

struct ChunkMemoryInfo {
  int64_t chunk_id;
  int64_t chunk_mem_size;
  std::vector<int64_t> mem_block_ids;
  ChunkMemoryInfo() : chunk_id(-1), chunk_mem_size(-1) {}
};

struct RankDeviceMemoryInfo {
  int64_t rank_id;
  int64_t device_id;
  ChunkMemoryInfo chunk_info;
  int64_t total_mem_size;
  int64_t not_reused_mem_size;
  std::vector<int64_t> not_reused_mem_block_ids;
  int64_t eager_variable_total_mem_size;
  std::vector<int64_t> eager_variable_mem_block_ids;
  RankDeviceMemoryInfo()
      : rank_id(-1),
        device_id(-1),
        total_mem_size(0),
        not_reused_mem_size(0),
        eager_variable_total_mem_size(0) {}
};

}  // namespace

void PlanUtil::PlanMemoryLog(Plan* plan, const std::string& plan_name) {
  std::vector<const TaskProto*> ordered_tasks;
  for (const TaskProto& task : plan->task()) { ordered_tasks.push_back(&task); }
  auto CompTask = [](const TaskProto* a, const TaskProto* b) {
    return a->task_set_info().order_in_graph() < b->task_set_info().order_in_graph();
  };
  std::sort(ordered_tasks.begin(), ordered_tasks.end(), CompTask);

  std::vector<RankDeviceMemoryInfo> rank_device_memory_infos(GlobalProcessCtx::WorldSize(),
                                                             RankDeviceMemoryInfo());
  HashMap<int64_t, MemBlockMemoryInfo> mem_block_id2info;
  HashMap<int64_t, const RegstDescProto*> regst_desc_id2regst;

  for (const ChunkProto& chunk : plan->block_chunk_list().chunk()) {
    int64_t rank_id = chunk.machine_id();
    auto& info = rank_device_memory_infos[rank_id];
    info.rank_id = rank_id;
    if (!memory::IsHostMem(chunk.mem_case())) { info.device_id = chunk.mem_case().device_id(); }
    info.total_mem_size += chunk.mem_size();
    info.chunk_info.chunk_id = chunk.chunk_id();
    info.chunk_info.chunk_mem_size = chunk.mem_size();
  }

  for (const MemBlockProto& mem_block : plan->block_chunk_list().mem_block()) {
    int64_t mem_block_id = mem_block.mem_block_id();
    mem_block_id2info.emplace(mem_block_id, MemBlockMemoryInfo());
    auto& info = mem_block_id2info.at(mem_block_id);
    info.mem_block_id = mem_block_id;
    info.mem_block_mem_size = mem_block.mem_size();
    auto& rank_memory_info = rank_device_memory_infos.at(mem_block.machine_id());
    if (!memory::IsHostMem(mem_block.mem_case())) {
      if (mem_block.has_chunk_id()) {
        rank_memory_info.chunk_info.mem_block_ids.push_back(mem_block_id);
      } else {
        if (mem_block.has_variable_op_name()) {
          rank_memory_info.eager_variable_mem_block_ids.push_back(mem_block_id);
          rank_memory_info.eager_variable_total_mem_size += mem_block.mem_size();
        } else {
          rank_memory_info.not_reused_mem_block_ids.push_back(mem_block_id);
          rank_memory_info.not_reused_mem_size += mem_block.mem_size();
        }
        rank_memory_info.total_mem_size += mem_block.mem_size();
      }
    }
  }

  for (const auto* task : ordered_tasks) {
    for (const auto& pair : task->produced_regst_desc()) {
      const auto& regst = pair.second;
      if (regst.regst_desc_type().has_data_regst_desc()
          && mem_block_id2info.find(regst.mem_block_id()) != mem_block_id2info.end()) {
        mem_block_id2info.at(regst.mem_block_id())
            .ordered_regst_desc_id.push_back(regst.regst_desc_id());
        regst_desc_id2regst.emplace(regst.regst_desc_id(), &regst);
      }
    }
  }

  auto CompMemBlock = [&](int64_t a, int64_t b) {
    return mem_block_id2info[a].mem_block_mem_size > mem_block_id2info[b].mem_block_mem_size;
  };

  auto B2MiB = [](int64_t val) { return val * 1.0 / 1000000.0; };

  for (auto& rank_memory_info : rank_device_memory_infos) {
    std::sort(rank_memory_info.chunk_info.mem_block_ids.begin(),
              rank_memory_info.chunk_info.mem_block_ids.end(), CompMemBlock);
    std::sort(rank_memory_info.not_reused_mem_block_ids.begin(),
              rank_memory_info.not_reused_mem_block_ids.end(), CompMemBlock);
    std::sort(rank_memory_info.eager_variable_mem_block_ids.begin(),
              rank_memory_info.eager_variable_mem_block_ids.end(), CompMemBlock);
    LOG(INFO) << "\n Graph name " << plan_name << " in Rank: " << rank_memory_info.rank_id
              << ", Device: " << rank_memory_info.device_id << " needs to allocate [ "
              << B2MiB(rank_memory_info.total_mem_size)
              << " MiB ] device memory. \n   In general, Chunk id: "
              << rank_memory_info.chunk_info.chunk_id << "  memory is [ "
              << B2MiB(rank_memory_info.chunk_info.chunk_mem_size)
              << " MiB ] with mem_block_num = " << rank_memory_info.chunk_info.mem_block_ids.size()
              << "\n        Unreused memory not eager var is  [ "
              << B2MiB(rank_memory_info.not_reused_mem_size)
              << " MiB ] with mem_block_num = " << rank_memory_info.not_reused_mem_block_ids.size()
              << "\n        Eager Variable Tensor total memory is [ "
              << B2MiB(rank_memory_info.eager_variable_total_mem_size)
              << " MiB ] with mem_block_num = "
              << rank_memory_info.eager_variable_mem_block_ids.size() << "\n";
  }

  auto Vlog3ForMemBlockDetails = [&](int64_t device_id, const std::vector<int64_t>& mem_block_ids,
                                     const std::string& prefix) {
    for (int64_t mem_block_id : mem_block_ids) {
      CHECK(mem_block_id2info.find(mem_block_id) != mem_block_id2info.end());
      const auto& mem_block_info = mem_block_id2info.at(mem_block_id);
      if (mem_block_info.ordered_regst_desc_id.size() != 1) { continue; }
      const auto* regst = regst_desc_id2regst.at(mem_block_info.ordered_regst_desc_id.at(0));
      const auto& data_regst = regst->regst_desc_type().data_regst_desc();
      const auto& lbi2blob_desc_pair = data_regst.lbi2blob_desc(0);
      std::string tensor_name = GenLogicalBlobName(lbi2blob_desc_pair.lbi());
      const auto& blob_desc = lbi2blob_desc_pair.blob_desc();
      VLOG(3) << "In Device: " << device_id << " Memblock id: " << mem_block_id << prefix
              << " size: " << B2MiB(mem_block_info.mem_block_mem_size)
              << " MiB, name: " << tensor_name << "\nshape: " << Shape(blob_desc.shape()).ToString()
              << " ,dtype: " << DataType_Name(blob_desc.data_type());
    }
  };

  for (const auto& rank_memory_info : rank_device_memory_infos) {
    int64_t chunk_id = rank_memory_info.chunk_info.chunk_id;
    int64_t device_id = rank_memory_info.device_id;
    VLOG(2) << "========================= "
            << "In Device : " << device_id << " Chunk Memory info details:";
    for (int64_t mem_block_id : rank_memory_info.chunk_info.mem_block_ids) {
      CHECK(mem_block_id2info.find(mem_block_id) != mem_block_id2info.end());
      const auto& mem_block_info = mem_block_id2info.at(mem_block_id);
      VLOG(2) << "     In Device: " << device_id << " Chunk id: " << chunk_id
              << " MemBlock id: " << mem_block_id
              << " has num = " << mem_block_info.ordered_regst_desc_id.size()
              << " tensor with mem size = " << B2MiB(mem_block_info.mem_block_mem_size);
      for (int64_t i = 0; i < mem_block_info.ordered_regst_desc_id.size(); ++i) {
        const auto* regst = regst_desc_id2regst.at(mem_block_info.ordered_regst_desc_id.at(i));
        const auto& data_regst = regst->regst_desc_type().data_regst_desc();
        const auto& lbi2blob_desc_pair = data_regst.lbi2blob_desc(0);
        std::string tensor_name = GenLogicalBlobName(lbi2blob_desc_pair.lbi());
        const auto& blob_desc = lbi2blob_desc_pair.blob_desc();
        std::string alloc_order = "inplaced";
        if (regst->has_alloc_before_actor()) {
          alloc_order = std::to_string(regst->alloc_before_actor());
        }
        std::string free_order = "inplaced";
        if (regst->has_free_after_actor()) {
          free_order = std::to_string(regst->free_after_actor());
        }
        VLOG(3) << "In Chunk id: " << chunk_id << ", MemBlock id: " << mem_block_id
                << " Order: " << i
                << " ,duration: " << (regst->free_after_actor() - regst->alloc_before_actor() + 1)
                << " ,size: " << B2MiB(BlobDesc(blob_desc).AlignedTotalByteSize())
                << " MiB, name: " << tensor_name
                << "\nshape: " << Shape(blob_desc.shape()).ToString()
                << " ,dtype: " << DataType_Name(blob_desc.data_type())
                << " ,alloc_order: " << alloc_order << " ,free_order: " << free_order;
      }
    }

    Vlog3ForMemBlockDetails(device_id, rank_memory_info.not_reused_mem_block_ids, " Unreused ");
    Vlog3ForMemBlockDetails(device_id, rank_memory_info.eager_variable_mem_block_ids,
                            " EagerVariable ");
  }
}

void PlanUtil::GenLightPlan(Plan* plan, const std::string& plan_name) {
  std::vector<const TaskProto*> ordered_tasks;
  for (const TaskProto& task : plan->task()) { ordered_tasks.push_back(&task); }
  auto CompTask = [](const TaskProto* a, const TaskProto* b) {
    return a->task_set_info().order_in_graph() < b->task_set_info().order_in_graph();
  };
  std::sort(ordered_tasks.begin(), ordered_tasks.end(), CompTask);

  HashMap<int64_t, std::string> task_id2name;
  HashMap<int64_t, const TaskProto*> task_id2proto;
  HashMap<int64_t, std::string> regst_id2name;
  HashMap<int64_t, const RegstDescProto&> regst_id2proto;
  for (const auto* task : ordered_tasks) {
    const auto& exec_seq = task->exec_sequence();
    std::string name;
    if (exec_seq.exec_node_size() >= 1) {
      const auto& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
      if (kernel_conf.has_op_attribute_ref()) {
        name = kernel_conf.op_attribute_ref();
      } else {
        name = kernel_conf.op_attribute().op_conf().name();
      }
    } else {
      name = TaskType_Name(task->task_type());
    }
    task_id2name.emplace(task->task_id(), name);
    task_id2proto.emplace(task->task_id(), task);
    CHECK(!name.empty());
    for (const auto& pair : task->produced_regst_desc()) {
      std::string regst_name = name + "/" + pair.first;
      regst_id2name.emplace(pair.second.regst_desc_id(), regst_name);
      regst_id2proto.emplace(pair.second.regst_desc_id(), pair.second);
    }
  }

  auto RegstId2TensorStr = [&](int64_t regst_id) -> std::string {
    CHECK(regst_id2proto.find(regst_id) != regst_id2proto.end())
        << " regst_id2proto cannot find: " << regst_id;
    std::ostringstream ss;
    ss << "{";
    const RegstDescProto& regst = regst_id2proto.at(regst_id);
    ss << "regust_num: " << std::to_string(regst.register_num());
    ss << ", device: " << *CHECK_JUST(DeviceTag4DeviceType(regst.mem_case().device_type()));
    if (regst.regst_desc_type().has_data_regst_desc()) {
      const DataRegstDesc& data = regst.regst_desc_type().data_regst_desc();
      ss << ", time_shape: " << Shape(data.time_shape()).ToString();
      const BlobDescProto& blob = data.lbi2blob_desc(0).blob_desc();
      ss << ", shape: " << Shape(blob.shape()).ToString();
      ss << ", dtype: " << DataType_Name(blob.data_type());
    } else {
      ss << ", ctrl";
    }
    ss << "}";
    return ss.str();
  };
  std::vector<std::vector<const TaskProto*>> rank2ordered_task(GlobalProcessCtx::WorldSize(),
                                                               std::vector<const TaskProto*>());
  for (const auto* task : ordered_tasks) {
    CHECK_LT(task->machine_id(), rank2ordered_task.size());
    rank2ordered_task.at(task->machine_id()).push_back(task);
  }
  for (int64_t rank = 0; rank < GlobalProcessCtx::WorldSize(); ++rank) {
    auto file_stream =
        TeePersistentLogStream::Create(plan_name + "_rank_" + std::to_string(rank) + "_light_plan");
    file_stream << "rank : " << std::to_string(rank) << "\n";
    CHECK_LT(rank, rank2ordered_task.size());
    const auto& ordered_task_in_rank = rank2ordered_task.at(rank);
    for (int64_t i = 0; i < ordered_task_in_rank.size(); ++i) {
      CHECK_LT(i, ordered_task_in_rank.size());
      const auto* task = ordered_task_in_rank.at(i);
      int64_t task_id = task->task_id();
      CHECK(task_id2name.find(task_id) != task_id2name.end())
          << " task_id2name cannot find" << task_id;
      int64_t thrd_id = task->thrd_id();
      StreamId stream_id = DecodeStreamIdFromInt64(thrd_id);
      file_stream << "order : " << std::to_string(i) << " , actor id : " << std::to_string(task_id)
                  << " name : " << task_id2name.at(task_id) << " thrd : " << std::to_string(thrd_id)
                  << " device_type : " << DeviceType_Name(stream_id.device_type())
                  << " stream_index : " << std::to_string(stream_id.stream_index()) << " {\n";
      for (const auto& key2consume_regst : task->consumed_regst_desc_id()) {
        std::string key = key2consume_regst.first;
        for (int64_t consume_regst_id : key2consume_regst.second.regst_desc_id()) {
          std::string other_rank_str = "";
          CHECK(regst_id2proto.find(consume_regst_id) != regst_id2proto.end())
              << " regst_id2proto cannot find: " << consume_regst_id;
          int64_t consume_task_id = regst_id2proto.at(consume_regst_id).producer_task_id();
          CHECK(task_id2proto.find(consume_task_id) != task_id2proto.end())
              << " task_id2proto cannot find: " << consume_task_id;
          int64_t other_rank = task_id2proto.at(consume_task_id)->machine_id();
          if (other_rank != rank) { other_rank_str = " , rank: " + std::to_string(other_rank); }
          CHECK(regst_id2name.find(consume_regst_id) != regst_id2name.end())
              << " regst_id2name cannot find: " << consume_regst_id;
          file_stream << "  consume : " << key << " : <- [ " << regst_id2name.at(consume_regst_id)
                      << " ] ( actor_id: " << std::to_string(consume_task_id) << other_rank_str
                      << ", regst: " << RegstId2TensorStr(consume_regst_id) << " )\n";
        }
      }
      for (const auto& key2produce_regst : task->produced_regst_desc()) {
        const RegstDescProto& regst = key2produce_regst.second;
        file_stream << "  produce : " << key2produce_regst.first
                    << " regst: " << RegstId2TensorStr(regst.regst_desc_id()) << " {\n";
        for (int64_t consumer_task_id : regst.consumer_task_id()) {
          std::string other_rank_str = "";
          CHECK(task_id2proto.find(consumer_task_id) != task_id2proto.end())
              << " task_id2proto cannot find " << consumer_task_id;
          CHECK(task_id2name.find(consumer_task_id) != task_id2name.end())
              << " task_id2name cannot find " << consumer_task_id;
          int64_t other_rank = task_id2proto.at(consumer_task_id)->machine_id();
          if (other_rank != rank) { other_rank_str = " , rank: " + std::to_string(other_rank); }
          file_stream << "    -> [ " << task_id2name.at(consumer_task_id)
                      << " ] ( actor_id: " << std::to_string(consumer_task_id) << other_rank_str
                      << " )\n";
        }
        file_stream << "  }\n";
      }

      file_stream << "}\n";
    }
  }
}

const oneflow::OpAttribute& PlanUtil::GetOpAttribute(const Plan* plan, int64_t job_id,
                                                     const oneflow::KernelConf& kernel_conf) {
  if (kernel_conf.has_op_attribute()) {
    return kernel_conf.op_attribute();
  } else if (kernel_conf.has_op_attribute_ref()) {
    auto table_it = plan->job_id2op_attribute_ref_table().find(job_id);
    CHECK(table_it != plan->job_id2op_attribute_ref_table().end())
        << "op attribute ref table not found for job id: " << job_id;
    ;
    auto it = table_it->second.op_name2op_attribute().find(kernel_conf.op_attribute_ref());
    CHECK(it != table_it->second.op_name2op_attribute().end())
        << "op attribute ref: " << kernel_conf.op_attribute_ref() << " not found";
    return it->second;
  } else {
    UNIMPLEMENTED() << "kernel_conf must has either op_attribute or op_attribute_ref. kernel_conf: "
                    << kernel_conf.DebugString();
  }
}

void PlanUtil::PopulateOpAttribute(
    Plan* plan,
    const PbMap<int64_t, ::oneflow::OpAttributeRefTable>& job_id2op_attribute_ref_table) {
  for (auto& task : *plan->mutable_task()) {
    if (task.exec_sequence().exec_node_size() == 1
        && task.exec_sequence().exec_node(0).kernel_conf().has_op_attribute_ref()) {
      auto* kernel_conf = task.mutable_exec_sequence()->mutable_exec_node(0)->mutable_kernel_conf();
      auto table_it = job_id2op_attribute_ref_table.find(task.job_id());
      CHECK(table_it != job_id2op_attribute_ref_table.end())
          << "op attribute ref table not found for job id: " << task.job_id();
      auto it = table_it->second.op_name2op_attribute().find(kernel_conf->op_attribute_ref());
      CHECK(it != table_it->second.op_name2op_attribute().end())
          << "ref: " << kernel_conf->op_attribute_ref() << " not found";
      *kernel_conf->mutable_op_attribute() = it->second;
      kernel_conf->clear_op_attribute_ref();
    } else {
      for (auto& exec_node : task.exec_sequence().exec_node()) {
        CHECK(exec_node.kernel_conf().has_op_attribute())
            << "op_attribute absent, exec_node: " << exec_node.DebugString();
      }
    }
  }
}

/*static*/ StreamId PlanUtil::GetStreamId(const TaskProto& task) {
  return DecodeStreamIdFromInt64(task.thrd_id());
}

/*static*/ int64_t PlanUtil::GetDeviceIndex(const TaskProto& task) {
  return GetStreamId(task).device_id().device_index();
}

}  // namespace oneflow
