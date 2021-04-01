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
#include "oneflow/core/job/inter_job_mem_sharing_util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

namespace {

void GetOpName2JobId2TaskProtos(
    Plan* plan, const HashSet<std::string>& op_names,
    HashMap<std::string, HashMap<int64_t, std::vector<TaskProto*>>>* op_name2job_id2task_protos) {
  for (int64_t i = 0; i < plan->task_size(); ++i) {
    TaskProto* task = plan->mutable_task(i);
    if (task->exec_sequence().exec_node_size() == 1) {
      const KernelConf& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
      std::string op_name =
          PlanUtil::GeOpAttribute(plan, task->job_id(), kernel_conf).op_conf().name();
      if (op_names.find(op_name) != op_names.end()) {
        CHECK(task->has_parallel_ctx());
        (*op_name2job_id2task_protos)[op_name][task->job_id()].push_back(task);
      }
    }
  }
  for (auto& op2job_task_pair : *op_name2job_id2task_protos) {
    for (auto& job2task_pair : op2job_task_pair.second) {
      std::vector<TaskProto*>& task_protos = job2task_pair.second;
      std::sort(task_protos.begin(), task_protos.end(),
                [](const TaskProto* lhs, const TaskProto* rhs) {
                  return lhs->parallel_ctx().parallel_id() < rhs->parallel_ctx().parallel_id();
                });
    }
  }
}

HashMap<std::string, HashSet<int64_t>> GetInterfaceOpName2JobIds(
    const std::vector<std::shared_ptr<Job>>& jobs) {
  HashMap<std::string, HashSet<int64_t>> interface_op_name2job_ids;
  HashSet<std::string> unique_op_name_check;
  FOR_RANGE(int64_t, i, 0, jobs.size()) {
    const auto& job = *jobs.at(i);
    for (const auto& op : job.net().op()) {
      if (IsInterfaceOpConf(op)) {
        CHECK(interface_op_name2job_ids[op.name()].emplace(i).second);
        unique_op_name_check.emplace(op.name());
      } else {
        // interface ops shouldn't share op_name with other ops
        CHECK(unique_op_name_check.find(op.name()) == unique_op_name_check.end());
      }
    }
  }
  return interface_op_name2job_ids;
}

std::vector<HashSet<int64_t>> InitJobId2MutualExclusionJobIds(
    const std::vector<std::shared_ptr<Job>>& jobs) {
  int64_t job_size = jobs.size();
  std::vector<HashSet<int64_t>> job_id2mutual_exclusion_ids(job_size);
  for (const auto& pair : GetInterfaceOpName2JobIds(jobs)) {
    for (int64_t first_id : pair.second) {
      for (int64_t second_id : pair.second) {
        if (first_id != second_id) { job_id2mutual_exclusion_ids[first_id].emplace(second_id); }
      }
    }
  }
  const InterJobReuseMemStrategy* strategy = Global<const InterJobReuseMemStrategy>::Get();
  if (strategy->has_custom_parallelism()) {
    auto* job_name2job_id = Global<JobName2JobId>::Get();
    for (const auto& group : strategy->custom_parallelism().nonparallel_group()) {
      for (const std::string& first_name : group.job_name()) {
        for (const std::string& second_name : group.job_name()) {
          if (first_name != second_name) {
            CHECK(job_name2job_id->find(first_name) != job_name2job_id->end());
            CHECK(job_name2job_id->find(second_name) != job_name2job_id->end());
            int64_t first_id = (*job_name2job_id)[first_name];
            int64_t second_id = (*job_name2job_id)[second_name];
            job_id2mutual_exclusion_ids[first_id].emplace(second_id);
          }
        }
      }
    }
  }
  return job_id2mutual_exclusion_ids;
}

std::vector<HashSet<int64_t>> GetMutualExclusionJobGroups(
    const std::vector<std::shared_ptr<Job>>& jobs) {
  int64_t job_size = jobs.size();
  std::vector<HashSet<int64_t>> job_groups;
  if (Global<const InterJobReuseMemStrategy>::Get()->has_reuse_mem_priority()) {
    job_groups.push_back(HashSet<int64_t>());
    FOR_RANGE(int64_t, i, 0, job_size) { job_groups.front().emplace(i); }
    return job_groups;
  }

  // default using parallelism_priority strategy
  std::vector<HashSet<int64_t>> job_id2mutual_exclusion_ids = InitJobId2MutualExclusionJobIds(jobs);
  std::vector<HashSet<int64_t>> job_id2enable_parallel_ids(job_size);
  FOR_RANGE(int64_t, i, 0, job_size) {
    FOR_RANGE(int64_t, j, 0, job_size) {
      if (job_id2mutual_exclusion_ids[i].find(j) == job_id2mutual_exclusion_ids[i].end()) {
        job_id2enable_parallel_ids[i].emplace(j);
      }
    }
  }
  int64_t mem_share_group_num = 0;
  std::vector<int64_t> job_id2mem_share_group_id(job_size, -1);
  FOR_RANGE(int64_t, this_job_id, 0, job_size) {
    HashSet<int64_t> mem_share_group_id_used;
    for (int64_t enable_parallel_job_id : job_id2enable_parallel_ids[this_job_id]) {
      int64_t group_id = job_id2mem_share_group_id[enable_parallel_job_id];
      if (group_id != -1) { mem_share_group_id_used.emplace(group_id); }
    }
    FOR_RANGE(int64_t, this_group_id, 0, mem_share_group_num) {
      if (mem_share_group_id_used.find(this_group_id) == mem_share_group_id_used.end()) {
        job_id2mem_share_group_id[this_job_id] = this_group_id;
        break;
      }
    }
    if (job_id2mem_share_group_id[this_job_id] == -1) {
      job_id2mem_share_group_id[this_job_id] = mem_share_group_num;
      ++mem_share_group_num;
      CHECK_LE(mem_share_group_num, job_size);
    }
  }

  job_groups.resize(mem_share_group_num);
  FOR_RANGE(int64_t, this_job_id, 0, job_size) {
    job_groups[job_id2mem_share_group_id[this_job_id]].emplace(this_job_id);
  }
  {
    HashSet<int64_t> job_id_unique_check;
    for (auto& job_group : job_groups) {
      for (int64_t job_id : job_group) { CHECK(job_id_unique_check.emplace(job_id).second); }
    }
  }
  return job_groups;
}

void MergeReusedChunk(HashMap<int64_t, ChunkProto>* chunk_id2chunk,
                      HashMap<int64_t, MemBlockProto*>* mem_block_id2mem_block,
                      const std::vector<HashSet<int64_t>>& reuse_mem_job_groups) {
  // mzuid = memory zone unique id
  HashMap<int64_t, HashMap<int64_t, int64_t>> job_id2mzuid2chunk_id;
  HashMap<int64_t, HashSet<MemBlockProto*>> chunk_id2mem_blocks;

  for (auto& pair : *mem_block_id2mem_block) {
    MemBlockProto* mem_block = pair.second;
    if (mem_block->enable_reuse_mem() == false) {
      CHECK(mem_block->has_chunk_id() == false);
      CHECK(mem_block->has_chunk_offset() == false);
      continue;
    }
    CHECK(mem_block->has_chunk_id() && mem_block->chunk_id() >= 0);
    CHECK(mem_block->has_chunk_offset() && mem_block->chunk_offset() >= 0);
    CHECK(chunk_id2mem_blocks[mem_block->chunk_id()].insert(mem_block).second);
  }

  // merge chunk and delete useless chunk
  for (const auto& pair : *chunk_id2chunk) {
    const ChunkProto& chunk = pair.second;
    const MemoryCase& mem_case = chunk.mem_case();
    // only reused mem in cuda device
    if (mem_case.has_host_mem()) { continue; }
    int64_t mzuid = MemoryCaseUtil::GenMemZoneUniqueId(chunk.machine_id(), mem_case);
    CHECK_EQ(chunk.job_id_size(), 1);
    CHECK(job_id2mzuid2chunk_id[chunk.job_id(0)].emplace(mzuid, chunk.chunk_id()).second);
  }

  auto MergeMemChunkIdR2L = [&](int64_t left_chunk_id, int64_t right_chunk_id) {
    CHECK_NE(left_chunk_id, right_chunk_id);
    ChunkProto* chunk_l = &(chunk_id2chunk->at(left_chunk_id));
    ChunkProto* chunk_r = &(chunk_id2chunk->at(right_chunk_id));
    CHECK_GE(chunk_l->job_id_size(), 1);
    CHECK_EQ(chunk_r->job_id_size(), 1);
    CHECK_EQ(chunk_l->machine_id(), chunk_r->machine_id());
    CHECK(chunk_l->mem_case() == chunk_r->mem_case());
    CHECK_GT(chunk_l->mem_size(), 0);
    CHECK_GT(chunk_r->mem_size(), 0);
    for (MemBlockProto* mem_block : chunk_id2mem_blocks[right_chunk_id]) {
      CHECK_EQ(mem_block->machine_id(), chunk_l->machine_id());
      CHECK(mem_block->mem_case() == chunk_l->mem_case());
      mem_block->set_chunk_id(left_chunk_id);
    }
    chunk_l->add_job_id(chunk_r->job_id(0));
    chunk_l->set_mem_size(std::max(chunk_l->mem_size(), chunk_r->mem_size()));
    chunk_id2chunk->erase(chunk_id2chunk->find(right_chunk_id));
  };
  auto InitMzuid2JobIdsInJobGroup =
      [&](const HashSet<int64_t>& job_group) -> HashMap<int64_t, HashSet<int64_t>> {
    HashMap<int64_t, HashSet<int64_t>> mzuid2job_ids;
    for (int64_t job_id : job_group) {
      for (const auto& pair : job_id2mzuid2chunk_id[job_id]) {
        CHECK(mzuid2job_ids[pair.first].emplace(job_id).second);
      }
    }
    return mzuid2job_ids;
  };
  for (const HashSet<int64_t>& job_group : reuse_mem_job_groups) {
    if (job_group.size() <= 1) { continue; }
    HashMap<int64_t, HashSet<int64_t>> mzuid2job_ids = InitMzuid2JobIdsInJobGroup(job_group);
    for (const auto& pair : mzuid2job_ids) {
      const HashSet<int64_t>& job_ids = pair.second;
      if (job_ids.size() <= 1) { continue; }
      int64_t mzuid = pair.first;
      int64_t merged_job_id = *(job_ids.begin());
      for (int64_t job_id : job_ids) {
        if (job_id == merged_job_id) { continue; }
        MergeMemChunkIdR2L(job_id2mzuid2chunk_id[merged_job_id].at(mzuid),
                           job_id2mzuid2chunk_id[job_id].at(mzuid));
      }
    }
  }
}

void MergeSharedMemBlockR2L(RegstDescProto* lhs, RegstDescProto* rhs,
                            HashMap<int64_t, MemBlockProto>* mem_block_id2mem_block) {
  if (lhs == rhs) { return; }
  auto CheckValidAndGetMemBlock = [&](int64_t mem_block_id, int64_t mem_size,
                                      const MemoryCase& mem_case) {
    CHECK_NE(mem_block_id, -1);
    CHECK(mem_block_id2mem_block->find(mem_block_id) != mem_block_id2mem_block->end());
    MemBlockProto* mem_block = &(mem_block_id2mem_block->at(mem_block_id));
    CHECK(mem_block->enable_reuse_mem() == false);
    CHECK(mem_block->has_chunk_id() == false);
    CHECK(mem_block->has_chunk_offset() == false);
    CHECK_EQ(mem_block->mem_size(), mem_size);
    CHECK(mem_block->mem_case() == mem_case);
    return mem_block;
  };

  auto MergeAndEraseMemBlock = [&](MemBlockProto* merged_block, MemBlockProto* erased_block) {
    CHECK_NE(merged_block->mem_block_id(), erased_block->mem_block_id());
    CHECK_EQ(erased_block->job_id_size(), 1);
    CHECK_EQ(merged_block->mem_size(), erased_block->mem_size());
    merged_block->add_job_id(erased_block->job_id(0));
    CHECK_EQ(mem_block_id2mem_block->erase(erased_block->mem_block_id()), 1);
  };

  int64_t merged_mem_block_id = lhs->mem_block_id();
  int64_t erased_mem_block_id = rhs->mem_block_id();
  CHECK(lhs->enable_reuse_mem() == false && rhs->enable_reuse_mem() == false);
  CHECK_EQ(lhs->mem_block_offset(), 0);
  CHECK_EQ(rhs->mem_block_offset(), 0);
  RtRegstDesc left_rt_regst(*lhs);
  RtRegstDesc right_rt_regst(*rhs);
  MemBlockProto* merged_mem_block = CheckValidAndGetMemBlock(
      merged_mem_block_id, left_rt_regst.TotalMainByteSize4AllRegst(), lhs->mem_case());
  MemBlockProto* erased_mem_block = CheckValidAndGetMemBlock(
      erased_mem_block_id, right_rt_regst.TotalMainByteSize4AllRegst(), rhs->mem_case());
  MergeAndEraseMemBlock(merged_mem_block, erased_mem_block);
  rhs->set_mem_block_id(merged_mem_block_id);

  int64_t separated_header_mem_size = left_rt_regst.TotalSeparatedHeaderByteSize4AllRegst();
  if (separated_header_mem_size > 0) {
    CHECK_EQ(separated_header_mem_size, right_rt_regst.TotalSeparatedHeaderByteSize4AllRegst());
    int64_t merged_header_id = lhs->separated_header_mem_block_id();
    int64_t erased_header_id = rhs->separated_header_mem_block_id();
    MemoryCase header_mem_case =
        MemoryCaseUtil::GetHostPinnedMemoryCaseForRegstSeparatedHeader(lhs->mem_case());
    MemBlockProto* merged_header_block =
        CheckValidAndGetMemBlock(merged_header_id, separated_header_mem_size, header_mem_case);
    MemBlockProto* erased_header_block =
        CheckValidAndGetMemBlock(erased_header_id, separated_header_mem_size, header_mem_case);
    MergeAndEraseMemBlock(merged_header_block, erased_header_block);
    rhs->set_separated_header_mem_block_id(merged_header_id);
  }
}

void MergeSharedInterfaceMemBlock(const std::vector<std::shared_ptr<Job>>& jobs, Plan* plan,
                                  HashMap<int64_t, MemBlockProto>* mem_block_id2mem_block) {
  HashMap<std::string, HashSet<int64_t>> interface_op_name2job_ids =
      GetInterfaceOpName2JobIds(jobs);
  HashSet<std::string> interface_op_names;
  for (const auto& pair : interface_op_name2job_ids) { interface_op_names.insert(pair.first); }
  HashMap<std::string, HashMap<int64_t, std::vector<TaskProto*>>> op_name2job_id2task_protos;
  GetOpName2JobId2TaskProtos(plan, interface_op_names, &op_name2job_id2task_protos);

  for (const auto& op_job_pair : interface_op_name2job_ids) {
    if (op_job_pair.second.size() <= 1) { continue; }
    const HashMap<int64_t, std::vector<TaskProto*>>& job_id2same_op_name_sorted_task_protos =
        op_name2job_id2task_protos.at(op_job_pair.first);
    const auto& first_vec = job_id2same_op_name_sorted_task_protos.begin()->second;
    std::vector<MemoryCase> common_mem_case_vec(first_vec.size());
    std::transform(
        first_vec.cbegin(), first_vec.cend(), common_mem_case_vec.begin(),
        [](TaskProto* tp) { return PlanUtil::GetSoleProducedDataRegst(tp)->mem_case(); });
    for (const auto& pair : job_id2same_op_name_sorted_task_protos) {
      const auto& task_protos = pair.second;
      CHECK_EQ(task_protos.size(), first_vec.size());
      FOR_RANGE(int64_t, i, 0, first_vec.size()) {
        CHECK_EQ(task_protos.at(i)->machine_id(), first_vec.at(i)->machine_id());
        RegstDescProto* first_regst_desc = PlanUtil::GetSoleProducedDataRegst(first_vec.at(i));
        RegstDescProto* regst_desc = PlanUtil::GetSoleProducedDataRegst(task_protos.at(i));

        MergeSharedMemBlockR2L(first_regst_desc, regst_desc, mem_block_id2mem_block);

        MemoryCase common_mem_case;
        CHECK(MemoryCaseUtil::GetCommonMemoryCase(common_mem_case_vec.at(i), regst_desc->mem_case(),
                                                  &common_mem_case));
        common_mem_case_vec[i] = common_mem_case;
      }
    }
    for (const auto& pair : job_id2same_op_name_sorted_task_protos) {
      const auto& task_protos = pair.second;
      FOR_RANGE(int64_t, i, 0, task_protos.size()) {
        RegstDescProto* regst_desc = PlanUtil::GetSoleProducedDataRegst(task_protos.at(i));
        *(regst_desc->mutable_mem_case()) = common_mem_case_vec.at(i);
        CHECK(mem_block_id2mem_block->find(regst_desc->mem_block_id())
              != mem_block_id2mem_block->end());
        *(mem_block_id2mem_block->at(regst_desc->mem_block_id()).mutable_mem_case()) =
            common_mem_case_vec.at(i);
      }
    }
  }
}

}  // namespace

void InterJobMemSharingUtil::MergeMemSharedInterfaceMemBlockBetweenJobs(
    const std::vector<std::shared_ptr<Job>>& jobs, Plan* plan) {
  if (jobs.size() == 1) { return; }

  HashMap<int64_t, MemBlockProto> mem_block_id2mem_block;
  for (const auto& mem_block : plan->block_chunk_list().mem_block()) {
    CHECK(mem_block_id2mem_block.emplace(mem_block.mem_block_id(), mem_block).second);
  }
  plan->mutable_block_chunk_list()->clear_mem_block();

  MergeSharedInterfaceMemBlock(jobs, plan, &mem_block_id2mem_block);

  for (const auto& pair : mem_block_id2mem_block) {
    *(plan->mutable_block_chunk_list()->add_mem_block()) = pair.second;
  }
}

void InterJobMemSharingUtil::MergeMemReusedChunkBetweenUserJobs(
    const std::vector<std::shared_ptr<Job>>& user_jobs, Plan* plan) {
  if (user_jobs.size() == 1) { return; }
  std::vector<HashSet<int64_t>> reuse_mem_job_groups = GetMutualExclusionJobGroups(user_jobs);

  HashMap<int64_t, ChunkProto> chunk_id2chunk;
  HashMap<int64_t, MemBlockProto*> mem_block_id2mem_block;
  for (const auto& chunk : plan->block_chunk_list().chunk()) {
    CHECK(chunk_id2chunk.emplace(chunk.chunk_id(), chunk).second);
  }
  plan->mutable_block_chunk_list()->clear_chunk();
  for (MemBlockProto& mem_block : *plan->mutable_block_chunk_list()->mutable_mem_block()) {
    CHECK(mem_block_id2mem_block.emplace(mem_block.mem_block_id(), &mem_block).second);
  }

  MergeReusedChunk(&chunk_id2chunk, &mem_block_id2mem_block, reuse_mem_job_groups);

  for (const auto& pair : chunk_id2chunk) {
    *(plan->mutable_block_chunk_list()->add_chunk()) = pair.second;
  }
}

}  // namespace oneflow
