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
#include "oneflow/core/job/improver.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/register/register_desc.pb.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/intra_job_mem_sharing_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/profiler.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/graph/plan_task_graph.h"
#include "oneflow/core/graph/sharable_mem_block_graph.h"
#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/blocking_counter.h"

namespace oneflow {

namespace {

double CalcRegstNum(double regst_desc_duration, double ii, double ii_scale) {
  return ((ii_scale - 1) * ii + regst_desc_duration) / (ii_scale * ii);
}

double CalcII(double regst_desc_duration, uint64_t regst_num, double ii_scale) {
  return regst_desc_duration / ((regst_num - 1) * ii_scale + 1);
}

uint64_t CalcRegstNum(
    const RegstDescProto& regst_desc,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    double ii,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId) {
  int64_t regst_desc_id = regst_desc.regst_desc_id();
  const auto& consumer_actor_id2duration = PathDurations4RegstDescId(regst_desc_id);
  const auto& consumer_actor_id2ii_scale = PathIIScales4RegstDescId(regst_desc_id);
  uint64_t regst_num = 0;
  for (const auto& pair : consumer_actor_id2duration) {
    double duration = pair.second;
    double ii_scale = consumer_actor_id2ii_scale.at(pair.first);
    uint64_t cur_path_regst_num = ceil(CalcRegstNum(duration, ii, ii_scale));
    regst_num = std::max(regst_num, cur_path_regst_num);
  }
  regst_num = std::max(regst_num, static_cast<uint64_t>(regst_desc.min_register_num()));
  regst_num = std::min(regst_num, static_cast<uint64_t>(regst_desc.max_register_num()));
  return regst_num;
}

uint64_t CalcMemoryConsumed(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
    double ii) {
  uint64_t mem_consuming = 0;
  HashMap<int64_t, uint64_t> mem_block_id2max_regst_desc_mem_bytes;
  for (const RegstDescProto* regst_desc : regst_descs) {
    uint64_t regst_num =
        CalcRegstNum(*regst_desc, PathDurations4RegstDescId, ii, PathIIScales4RegstDescId);
    uint64_t total_byte_size = RtRegstDesc(*regst_desc).TotalMainByteSize4AllRegst();
    if (regst_desc->mem_block_id() == -1) {
      mem_consuming += RoundUp(total_byte_size, kCudaMemAllocAlignSize);
    } else {
      total_byte_size += regst_desc->mem_block_offset();
      CHECK_EQ(regst_num, 1);
      int32_t mem_block_id = regst_desc->mem_block_id();
      auto& max_bytes = mem_block_id2max_regst_desc_mem_bytes[mem_block_id];
      max_bytes = std::max(max_bytes, total_byte_size);
    }
  }
  for (const auto& pair : mem_block_id2max_regst_desc_mem_bytes) {
    mem_consuming += RoundUp(pair.second, kCudaMemAllocAlignSize);
  }
  return mem_consuming;
}

std::function<uint64_t(int64_t)> MakeGetterGetPlanRegstNum(Plan* plan) {
  auto MutRestDesc4Id = PlanUtil::MakeMutRegstDesc4Id(plan);
  return [MutRestDesc4Id](int64_t regst_desc_id) {
    return MutRestDesc4Id(regst_desc_id)->register_num();
  };
}

std::function<void(int64_t, uint64_t)> MakeSetterSetPlanRegstNum(Plan* plan) {
  auto MutRestDesc4Id = PlanUtil::MakeMutRegstDesc4Id(plan);
  return [MutRestDesc4Id](int64_t regst_desc_id, uint64_t num) {
    MutRestDesc4Id(regst_desc_id)->set_register_num(num);
  };
}

std::function<const HashMap<int64_t, double>&(int64_t)> MakeGetterPathDurations4RegstDescId(
    const ChainActGraph& graph) {
  auto regst_desc_id2consumer_id2duration =
      std::make_shared<HashMap<int64_t, HashMap<int64_t, double>>>();
  graph.ForEachRegstDescConsumerPathMeanDuration(
      [&](int64_t regst_desc_id, int64_t consumer_actor_id, double time) {
        (*regst_desc_id2consumer_id2duration)[regst_desc_id][consumer_actor_id] = time;
      });
  auto empty = std::make_shared<const HashMap<int64_t, double>>();
  return [regst_desc_id2consumer_id2duration,
          empty](int64_t regst_desc_id) -> const HashMap<int64_t, double>& {
    const auto& it = regst_desc_id2consumer_id2duration->find(regst_desc_id);
    if (it == regst_desc_id2consumer_id2duration->end()) {
      return *empty;
    } else {
      return it->second;
    }
  };
}

std::function<const HashMap<int64_t, double>&(int64_t)> MakeGetterPathIIScales4RegstDescId(
    const ChainActGraph& graph) {
  auto regst_desc_id2consumer_id2ii_scale =
      std::make_shared<HashMap<int64_t, HashMap<int64_t, double>>>();
  graph.ForEachRegstDescConsumerPathIIScale(
      [&](int64_t regst_desc_id, int64_t consumer_actor_id, double ii_scale) {
        (*regst_desc_id2consumer_id2ii_scale)[regst_desc_id][consumer_actor_id] = ii_scale;
      });
  auto empty = std::make_shared<const HashMap<int64_t, double>>();
  return [regst_desc_id2consumer_id2ii_scale,
          empty](int64_t regst_desc_id) -> const HashMap<int64_t, double>& {
    const auto& it = regst_desc_id2consumer_id2ii_scale->find(regst_desc_id);
    if (it == regst_desc_id2consumer_id2ii_scale->end()) {
      return *empty;
    } else {
      return it->second;
    }
  };
}

void TryConnectWithMemSafeGuardCtrlRegstDesc(TaskProto* src_task_proto, TaskProto* dst_task_proto) {
  RegstDescProto* ctrl_regst_desc =
      FindOrCreateProducedCtrlRegstDesc(src_task_proto, "out_ctrl_shared_mem_safe_guard");
  int64_t dst_task_id = dst_task_proto->task_id();
  if (!IsInRepeatedField(ctrl_regst_desc->consumer_task_id(), dst_task_id)) {
    ctrl_regst_desc->add_consumer_task_id(dst_task_id);
    int64_t ctrl_regst_desc_id = ctrl_regst_desc->regst_desc_id();
    RegstDescIdSet* consumed_ctrl_regst_desc_ids =
        FindOrCreateConsumedCtrlRegstDescIdSet(dst_task_proto, "in_ctrl");
    CHECK(!IsInRepeatedField(consumed_ctrl_regst_desc_ids->regst_desc_id(), ctrl_regst_desc_id));
    consumed_ctrl_regst_desc_ids->add_regst_desc_id(ctrl_regst_desc_id);
    DumpToConsumedRegstDescId2Addr(*ctrl_regst_desc, dst_task_proto);
  }
}

void CollectTailRegstConsumerTaskIds(const std::vector<const RegstDescProto*>& shared_mem_regsts,
                                     HashSet<int64_t>* task_ids) {
  for (const RegstDescProto* regst_proto : shared_mem_regsts) {
    if (regst_proto == shared_mem_regsts.front()) { continue; }
    for (int64_t consumer_id : regst_proto->consumer_task_id()) { task_ids->insert(consumer_id); }
  }
}

void CollectSinkTaskIds(const HashSet<int64_t>& task_ids,
                        const std::function<bool(int64_t, int64_t)>& IsReachable,
                        std::list<int64_t>* sink_task_ids) {
  auto IsReachableToAnyOtherTask = [&](int64_t src_task_id) -> bool {
    for (int64_t dst_task_id : task_ids) {
      if (src_task_id == dst_task_id) { continue; }
      if (IsReachable(src_task_id, dst_task_id)) { return true; }
    }
    return false;
  };
  sink_task_ids->clear();
  for (int64_t src_task_id : task_ids) {
    if (!IsReachableToAnyOtherTask(src_task_id)) { sink_task_ids->push_back(src_task_id); }
  }
}

std::function<void(const std::vector<const RegstDescProto*>&)> MakeSetterAddCtrlRegst(
    Plan* plan, const std::function<bool(int64_t, int64_t)>& IsReachable) {
  auto task_id2task_proto = std::make_shared<HashMap<int64_t, TaskProto*>>();
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task_proto = plan->mutable_task(i);
    CHECK(task_id2task_proto->emplace(task_proto->task_id(), task_proto).second);
  }
  return [task_id2task_proto,
          IsReachable](const std::vector<const RegstDescProto*>& shared_mem_regsts) {
    if (shared_mem_regsts.size() == 1) { return; }
    int64_t header_task_id = shared_mem_regsts.front()->producer_task_id();
    TaskProto* header_task_proto = task_id2task_proto->at(header_task_id);
    HashSet<int64_t> tail_regsts_consumer_task_ids;
    CollectTailRegstConsumerTaskIds(shared_mem_regsts, &tail_regsts_consumer_task_ids);
    std::list<int64_t> sink_task_ids;
    CollectSinkTaskIds(tail_regsts_consumer_task_ids, IsReachable, &sink_task_ids);
    for (int64_t sink_task_id : sink_task_ids) {
      TaskProto* sink_task_proto = task_id2task_proto->at(sink_task_id);
      TryConnectWithMemSafeGuardCtrlRegstDesc(header_task_proto, sink_task_proto);
    }
  };
}

void FixReliantCtrlRegstNum(const Plan& plan, const std::function<uint64_t(int64_t)>& GetRegstNum,
                            const std::function<void(int64_t, uint64_t)>& SetRegstNum) {
  for (const auto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      const RegstDescProto& regst = pair.second;
      const RegstDescTypeProto& regst_type = regst.regst_desc_type();
      if (regst_type.has_ctrl_regst_desc()
          && regst_type.ctrl_regst_desc().has_reliant_regst_desc_id()) {
        // set ctrl regst num between copyHd and MdUpdt
        CHECK(task_proto.task_type() == kCopyHd);
        uint64_t regst_num = GetRegstNum(regst_type.ctrl_regst_desc().reliant_regst_desc_id());
        SetRegstNum(regst.regst_desc_id(), regst_num);
      }
    }
  }
}

void SetInplaceConsumedRegstDescId(Plan* plan,
                                   const std::function<RegstDescProto*(int64_t)>& RegstDesc4Id) {
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      RegstDescProto* regst_desc = &pair.second;
      CHECK_EQ(regst_desc->has_inplace_consumed_regst_desc_id(), false);
      if (regst_desc->has_hint_inplace_consumed_regst_desc_id()) {
        int64_t hint = regst_desc->hint_inplace_consumed_regst_desc_id();
        const RegstDescProto* in_regst_desc = RegstDesc4Id(hint);
        if (in_regst_desc->mem_block_id() != -1
            && in_regst_desc->mem_block_id() == regst_desc->mem_block_id()
            && in_regst_desc->mem_block_offset() == regst_desc->mem_block_offset()) {
          CHECK_EQ(in_regst_desc->register_num(), regst_desc->register_num());
          regst_desc->set_inplace_consumed_regst_desc_id(hint);
        }
      }
    }
  }
}

void SetUniqueMemBlockId4UnreusedMemRegst(Plan* plan) {
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      RegstDescProto* regst_desc = &pair.second;
      if (regst_desc->mem_block_id() == -1) {
        CHECK_EQ(regst_desc->mem_block_offset(), -1);
        regst_desc->set_mem_block_id(Global<IDMgr>::Get()->NewMemBlockId());
        regst_desc->set_mem_block_offset(0);
      }
    }
  }
}

void GenMemBlockAndChunk4Plan(Plan* plan) {
  HashMap<int64_t, MemBlockProto> mem_block_id2mem_block;
  // mzuid = memory zone unique id
  HashMap<int64_t, ChunkProto> mzuid2chunk;

  auto GenMemBlock4RegstIfNeed = [&](RegstDescProto* regst_desc, const TaskProto* task) {
    const int64_t job_id = task->job_id();
    const int64_t machine_id = task->machine_id();
    const int64_t thrd_id = task->thrd_id();
    int64_t mem_block_id = regst_desc->mem_block_id();
    int64_t mem_block_offset = regst_desc->mem_block_offset();
    CHECK_NE(mem_block_id, -1);
    CHECK_NE(mem_block_offset, -1);
    CHECK_EQ(regst_desc->separated_header_mem_block_id(), -1);

    RtRegstDesc rt_regst_desc(*regst_desc);
    int64_t regst_main_size = rt_regst_desc.TotalMainByteSize4AllRegst();
    int64_t regst_separated_size = rt_regst_desc.TotalSeparatedHeaderByteSize4AllRegst();

    if (mem_block_id2mem_block.find(mem_block_id) == mem_block_id2mem_block.end()) {
      MemBlockProto mem_block;
      mem_block.set_mem_block_id(mem_block_id);
      mem_block.add_job_id(job_id);
      mem_block.set_machine_id(machine_id);
      *(mem_block.mutable_mem_case()) = regst_desc->mem_case();
      mem_block.set_enable_reuse_mem(regst_desc->enable_reuse_mem());
      mem_block.set_mem_size(regst_main_size + mem_block_offset);
      mem_block.set_thrd_id_hint(thrd_id);
      CHECK(mem_block_id2mem_block.emplace(mem_block.mem_block_id(), mem_block).second);
    } else {
      MemBlockProto* mem_block = &(mem_block_id2mem_block.at(mem_block_id));
      CHECK_EQ(mem_block->job_id(0), job_id);
      CHECK_EQ(mem_block->machine_id(), machine_id);
      CHECK(mem_block->mem_case() == regst_desc->mem_case());
      CHECK_EQ(mem_block->enable_reuse_mem(), regst_desc->enable_reuse_mem());
      mem_block->set_mem_size(std::max(mem_block->mem_size(), regst_main_size + mem_block_offset));
    }

    if (regst_separated_size > 0) {
      int64_t separated_mem_block_id = Global<IDMgr>::Get()->NewMemBlockId();
      regst_desc->set_separated_header_mem_block_id(separated_mem_block_id);
      MemBlockProto mem_block;
      mem_block.set_mem_block_id(separated_mem_block_id);
      mem_block.add_job_id(job_id);
      mem_block.set_machine_id(machine_id);
      *(mem_block.mutable_mem_case()) =
          MemoryCaseUtil::GetHostPinnedMemoryCaseForRegstSeparatedHeader(regst_desc->mem_case());
      mem_block.set_enable_reuse_mem(false);
      mem_block.set_mem_size(regst_separated_size);
      mem_block.set_thrd_id_hint(thrd_id);
      CHECK(mem_block_id2mem_block.emplace(mem_block.mem_block_id(), mem_block).second);
    }
  };

  auto GenChunk4ReusedMemBlockIfNeed = [&](MemBlockProto* mem_block) {
    int64_t mzuid =
        MemoryCaseUtil::GenMemZoneUniqueId(mem_block->machine_id(), mem_block->mem_case());
    if (mzuid2chunk.find(mzuid) == mzuid2chunk.end()) {
      ChunkProto chunk;
      chunk.set_chunk_id(Global<IDMgr>::Get()->NewChunkId());
      chunk.add_job_id(mem_block->job_id(0));
      chunk.set_machine_id(mem_block->machine_id());
      *(chunk.mutable_mem_case()) = mem_block->mem_case();
      chunk.set_mem_size(mem_block->mem_size());
      CHECK(mzuid2chunk.emplace(mzuid, chunk).second);
      mem_block->set_chunk_id(chunk.chunk_id());
      mem_block->set_chunk_offset(0);
    } else {
      ChunkProto* chunk = &(mzuid2chunk.at(mzuid));
      CHECK_EQ(chunk->job_id(0), mem_block->job_id(0));
      mem_block->set_chunk_id(chunk->chunk_id());
      mem_block->set_chunk_offset(chunk->mem_size());
      chunk->set_mem_size(chunk->mem_size() + mem_block->mem_size());
    }
  };

  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      GenMemBlock4RegstIfNeed(&pair.second, task);
    }
  }

  for (auto& pair : mem_block_id2mem_block) {
    MemBlockProto* mem_block = &pair.second;
    CHECK(mem_block->has_chunk_id() == false);
    CHECK(mem_block->has_chunk_offset() == false);
    if (mem_block->enable_reuse_mem()) { GenChunk4ReusedMemBlockIfNeed(mem_block); }
  }

  for (const auto& pair : mem_block_id2mem_block) {
    *(plan->mutable_block_chunk_list()->add_mem_block()) = pair.second;
  }

  for (const auto& pair : mzuid2chunk) {
    *(plan->mutable_block_chunk_list()->add_chunk()) = pair.second;
  }
}

}  // namespace

uint64_t Improver::AvailableMemSize(int64_t machine_id, int64_t memory_zone_id) const {
  uint64_t mem_size = amd_.machine_amd(machine_id).zone_size(memory_zone_id);
  const ResourceDesc* resource_desc = Global<ResourceDesc, ForSession>::Get();
  const bool is_host = memory_zone_id == resource_desc->GpuDeviceNum();
  if (is_host) {
    mem_size -= resource_desc->reserved_host_mem_byte();
  } else {
    mem_size -= resource_desc->reserved_device_mem_byte();
  }
  CHECK_GT(mem_size, 0) << "memory_zone_id: " << memory_zone_id
                        << ", is_host: " << (is_host ? "yes" : "no") << "\n"
                        << "AvailableMemDesc:" << amd_.DebugString();
  return static_cast<uint64_t>(mem_size);
}

int64_t Improver::GetMemoryZoneId(const MemoryCase& mem_case) const {
  if (mem_case.has_device_cuda_mem()) {
    return mem_case.device_cuda_mem().device_id();
  } else {
    return Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum();
  }
}

void Improver::MakeMemZoneRegstDescs(const Plan& plan, MemZoneRegstDescs* mz2regst_desc) const {
  mz2regst_desc->resize(amd_.machine_amd_size());
  FOR_RANGE(int64_t, machine_id, 0, amd_.machine_amd_size()) {
    mz2regst_desc->at(machine_id).resize(amd_.machine_amd(machine_id).zone_size_size());
  }
  for (const auto& task : plan.task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      int64_t mem_zone_id = GetMemoryZoneId(pair.second.mem_case());
      mz2regst_desc->at(task.machine_id()).at(mem_zone_id).push_back(&pair.second);
    }
  }
}

Maybe<void> Improver::CheckAllZoneNotOOM(
    const MemZoneRegstDescs& mz_regst_descs,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
    double ii) const {
  FOR_RANGE(int64_t, machine_id, 0, mz_regst_descs.size()) {
    FOR_RANGE(int64_t, mem_zone_id, 0, mz_regst_descs[machine_id].size()) {
      const auto& regst_descs = mz_regst_descs[machine_id][mem_zone_id];
      const uint64_t calc =
          CalcMemoryConsumed(regst_descs, PathDurations4RegstDescId, PathIIScales4RegstDescId, ii);
      const uint64_t available = AvailableMemSize(machine_id, mem_zone_id);
      if (calc >= available) {
        const auto* id_mgr = Global<IDMgr>::Get();
        const std::string device_tag = *JUST(DeviceTag4DeviceType(
            id_mgr->IsGpuMemZone(mem_zone_id) ? DeviceType::kGPU : DeviceType::kCPU));
        return Error::MemoryZoneOutOfMemoryError(machine_id, mem_zone_id, calc, available,
                                                 device_tag)
               << "OOM detected at compile time. ";
      }
    }
  }
  return Maybe<void>::Ok();
}

double Improver::CalcMaxRegstDescDuration(
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const MemZoneRegstDescs& mz_regst_descs) const {
  double max_duration = 0;
  for (const auto& zone_regst_descs : mz_regst_descs) {
    for (const auto& regst_descs : zone_regst_descs) {
      for (const RegstDescProto* regst_desc : regst_descs) {
        for (const auto& pair : PathDurations4RegstDescId(regst_desc->regst_desc_id())) {
          max_duration = std::max(max_duration, pair.second);
        }
      }
    }
  }
  return max_duration;
}

Maybe<double> Improver::BinarySearchII(
    double base_ii,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
    const MemZoneRegstDescs& mz_regst_descs) const {
  double max_duration = CalcMaxRegstDescDuration(PathDurations4RegstDescId, mz_regst_descs);
  JUST(CheckAllZoneNotOOM(mz_regst_descs, PathDurations4RegstDescId, PathIIScales4RegstDescId,
                          max_duration));
  const double ii_search_threshold = 1;
  double r = max_duration;
  double l = base_ii;
  double mid = base_ii;
  while ((r - l) > ii_search_threshold) {
    mid = (l + r) / 2;
    const auto& oom_status = TRY(CheckAllZoneNotOOM(mz_regst_descs, PathDurations4RegstDescId,
                                                    PathIIScales4RegstDescId, mid));

    if (oom_status.IsOk()) {
      r = mid;
    } else if (oom_status.error()->has_memory_zone_out_of_memory_error()) {
      l = mid;
    } else {
      return oom_status.error();
    }
  }
  return r;
}

Maybe<void> Improver::ForEachImprovedRegstNum(
    const Plan& plan, bool is_memory_limited, double ii,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
    const std::function<void(int64_t, uint64_t)>& Handler) const {
  if (is_memory_limited) {
    MemZoneRegstDescs mz_regst_descs;
    MakeMemZoneRegstDescs(plan, &mz_regst_descs);
    ii = JUST(
        BinarySearchII(ii, PathDurations4RegstDescId, PathIIScales4RegstDescId, mz_regst_descs));
  }
  LOG(INFO) << "memory " << (is_memory_limited ? "limited" : "unlimited") << " ii: " << ii;
  for (const auto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      uint64_t regst_num = 0;
      if (pair.second.has_inplace_consumed_regst_desc_id()) {
        regst_num = pair.second.register_num();
      } else {
        regst_num =
            CalcRegstNum(pair.second, PathDurations4RegstDescId, ii, PathIIScales4RegstDescId);
      }
      Handler(pair.second.regst_desc_id(), regst_num);
    }
  }
  return Maybe<void>::Ok();
}

void Improver::Init(const AvailableMemDesc& amd, const Plan& naive_plan) {
  start_mem_block_id_ = Global<IDMgr>::Get()->NewMemBlockId();
  amd_ = amd;
}

}  // namespace oneflow
