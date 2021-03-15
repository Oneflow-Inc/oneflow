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
#include "oneflow/core/graph/id_serialization.h"
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
#include "oneflow/core/graph/regst_lifetime_graph.h"
#include "oneflow/core/graph/sharable_mem_block_graph.h"
#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/blocking_counter.h"

namespace oneflow {

namespace {

bool IsSharableRegstWithoutConsumer(const RegstDescProto& regst_desc) {
  return regst_desc.mem_block_id() == -1 && regst_desc.consumer_task_id_size() == 0
         && regst_desc.enable_reuse_mem();
}

bool IsConsumersAndProducerInSameChain(const RegstDescProto& regst_desc,
                                       const std::function<int64_t(int64_t)>& ChainId4TaskId) {
  int64_t producer_chain_id = ChainId4TaskId(regst_desc.producer_task_id());
  for (int64_t consumer_task_id : regst_desc.consumer_task_id()) {
    if (ChainId4TaskId(consumer_task_id) != producer_chain_id) { return false; }
  }
  return true;
}

void ForEachSharableStreamRegstDescsWithoutConsumer(
    const Plan& plan,
    const std::function<void(const std::vector<const RegstDescProto*>&)>& Handler) {
  HashMap<int64_t, std::vector<const RegstDescProto*>> global_work_stream_id2regst_descs;
  for (const auto& task : plan.task()) {
    int64_t global_work_stream_id = Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(task.task_id());
    for (const auto& pair : task.produced_regst_desc()) {
      if (IsSharableRegstWithoutConsumer(pair.second)) {
        global_work_stream_id2regst_descs[global_work_stream_id].push_back(&pair.second);
      }
    }
  }
  for (const auto& pair : global_work_stream_id2regst_descs) {
    if (pair.second.size() > 1) { Handler(pair.second); }
  }
}

void ForEachSameColoredStreamRegstDescWithoutConsumer(
    const Plan& plan,
    const std::function<void(const std::vector<const RegstDescProto*>&)>& Handler) {
  auto GetProducerTaskId = [](const RegstDescProto* regst_desc, HashSet<int64_t>* ret_actor_ids) {
    CHECK(regst_desc->enable_reuse_mem());
    ret_actor_ids->insert(regst_desc->producer_task_id());
  };
  ForEachSharableStreamRegstDescsWithoutConsumer(
      plan, [&](const std::vector<const RegstDescProto*>& regst_descs) {
        RegstLifetimeGraph(regst_descs, GetProducerTaskId).ForEachSameColoredRegstDescs(Handler);
      });
}

void ForEachSameColoredChainRegstDescs(
    const SharableMemBlockGraph& sharable_mem_block_gph,
    const std::function<std::vector<const RegstDescProto*>(
        const std::vector<const SharableMemBlockNode*>&)>& GetRegstDescs,
    const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>&
        ComputeLifetimeSameChainActorIds,
    const std::function<void(const std::vector<const RegstDescProto*>&)>& Handler) {
  std::vector<std::vector<const SharableMemBlockNode*>> sharable_mem_blocks_vec;
  sharable_mem_block_gph.ForEachSourceNodeGroup(
      &SharableMemBlockNode::chain_id,
      [&](const std::vector<const SharableMemBlockNode*>& sharable_mem_blocks) {
        sharable_mem_blocks_vec.push_back(sharable_mem_blocks);
      });
  std::vector<std::vector<std::vector<const RegstDescProto*>>> same_colored_regst_descs_vec(
      sharable_mem_blocks_vec.size());
  int64_t cpu_num = std::thread::hardware_concurrency();
  int64_t thread_pool_size = std::min<int64_t>(sharable_mem_blocks_vec.size(), cpu_num);
  BlockingCounter counter(sharable_mem_blocks_vec.size());
  ThreadPool thread_pool(thread_pool_size);
  FOR_RANGE(int64_t, i, 0, sharable_mem_blocks_vec.size()) {
    thread_pool.AddWork([i, &GetRegstDescs, &ComputeLifetimeSameChainActorIds,
                         &sharable_mem_blocks_vec, &same_colored_regst_descs_vec, &counter]() {
      const auto& sharable_mem_blocks = sharable_mem_blocks_vec.at(i);
      RegstLifetimeGraph(GetRegstDescs(sharable_mem_blocks), ComputeLifetimeSameChainActorIds)
          .ForEachSameColoredRegstDescs([&](const std::vector<const RegstDescProto*>& regst_descs) {
            same_colored_regst_descs_vec.at(i).push_back(regst_descs);
          });
      counter.Decrease();
    });
  }
  counter.WaitUntilCntEqualZero();
  for (const auto& regst_descs_vec : same_colored_regst_descs_vec) {
    for (const auto& regst_descs : regst_descs_vec) { Handler(regst_descs); }
  }
}

void ForEachSameColoredChainRegstDescWithConsumer(
    const PlanTaskGraph& plan_task_graph,
    const std::function<void(const std::vector<const RegstDescProto*>&)>& Handler) {
  // construct SharableMemBlockGraph
  auto ChainId4TaskId = [&](int64_t task_id) {
    return plan_task_graph.TaskProto4TaskId(task_id)->task_set_info().chain_id();
  };
  auto IsSharableRegstWithConsumer = [&](const RegstDescProto& regst_desc) {
    return regst_desc.mem_block_id() == -1 && regst_desc.consumer_task_id_size() > 0
           && regst_desc.enable_reuse_mem() && regst_desc.register_num() == 1
           && IsConsumersAndProducerInSameChain(regst_desc, ChainId4TaskId);
  };
  SharableMemBlockGraph sharable_mem_block_gph(plan_task_graph, IsSharableRegstWithConsumer);
  // group regst_descs for pre-colored regst_descs.
  // example:
  // given dlnet: A -> B -> C -> D -> E -> F -> H -> I, where D is a inplace op.
  // Regst(C) and Regst(D) are pre-colored with same color as a group, which
  // then shares memory with other regsts like A, B, E, ...
  HashMap<const RegstDescProto*, std::vector<const RegstDescProto*>> header2members;
  for (const SharableMemBlockNode* sharable_mem_block : sharable_mem_block_gph.source_nodes()) {
    auto regst_descs = sharable_mem_block->regst_descs();
    HashMap<const RegstDescProto*, size_t> regst_desc2mem_size;
    for (const RegstDescProto* regst_desc : regst_descs) {
      size_t size = RtRegstDesc(*regst_desc).TotalMainByteSize4AllRegst();
      CHECK(regst_desc2mem_size.emplace(regst_desc, size).second);
    }
    std::sort(regst_descs.begin(), regst_descs.end(),
              [&](const RegstDescProto* lhs, const RegstDescProto* rhs) {
                return regst_desc2mem_size.at(lhs) > regst_desc2mem_size.at(rhs);
              });
    header2members.emplace(regst_descs.at(0), regst_descs);
  }
  auto GetRegstDescs = [&](const std::vector<const SharableMemBlockNode*>& sharable_mem_blocks) {
    std::vector<const RegstDescProto*> ret;
    for (const SharableMemBlockNode* sharable_mem_block : sharable_mem_blocks) {
      for (const RegstDescProto* regst_desc : sharable_mem_block->regst_descs()) {
        if (header2members.find(regst_desc) != header2members.end()) {
          ret.push_back(regst_desc);
          break;
        }
      }
    }
    return ret;
  };
  auto ComputeLifetimeSameChainActorIds = [&](const RegstDescProto* regst_desc,
                                              HashSet<int64_t>* ret_actor_ids) {
    CHECK(regst_desc->enable_reuse_mem());
    ret_actor_ids->clear();
    for (const RegstDescProto* member : header2members.at(regst_desc)) {
      plan_task_graph.ComputeLifetimeSameChainActorIds(member, ret_actor_ids);
    }
  };
  auto AppendGroupMembers = [&](const std::vector<const RegstDescProto*>& regst_descs) {
    std::vector<const RegstDescProto*> members;
    for (const auto* header : regst_descs) {
      for (const auto* member : header2members.at(header)) { members.push_back(member); }
    }
    Handler(members);
  };
  ForEachSameColoredChainRegstDescs(sharable_mem_block_gph, GetRegstDescs,
                                    ComputeLifetimeSameChainActorIds, AppendGroupMembers);
}

void ForEachInferredMemBlockId(const PlanTaskGraph& plan_task_graph,
                               const std::function<void(int64_t, int64_t)>& Handler) {
  const Plan& plan = plan_task_graph.plan();
  auto HandleMemBlockId = [&](const std::vector<const RegstDescProto*>& regst_descs) {
    int64_t mem_block_id = Global<IDMgr>::Get()->NewMemBlockId();
    for (const RegstDescProto* regst_desc : regst_descs) {
      Handler(regst_desc->regst_desc_id(), mem_block_id);
    }
  };
  ForEachSameColoredStreamRegstDescWithoutConsumer(plan, HandleMemBlockId);
  ForEachSameColoredChainRegstDescWithConsumer(plan_task_graph, HandleMemBlockId);
}

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
  int64_t mem_size = amd_.machine_amd(machine_id).zone_size(memory_zone_id);
  const ResourceDesc* resource_desc = Global<ResourceDesc, ForSession>::Get();
  if (memory_zone_id == resource_desc->GpuDeviceNum()) {
    mem_size -= resource_desc->reserved_host_mem_byte();
  } else {
    mem_size -= resource_desc->reserved_device_mem_byte();
  }
  CHECK_GT(mem_size, 0);
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
        const std::string device_tag =
            *JUST(DeviceTag4DeviceType(DeserializeMemZoneIdFromInt64(mem_zone_id).device_type()));
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

void Improver::ForEachInferredMemBlockCriticalSection(
    const Plan& plan, const std::function<int64_t(int64_t)>& OrderInGraph4TaskId,
    const std::function<void(const std::vector<const RegstDescProto*>&)>& Handler) const {
  HashMap<int32_t, std::vector<const RegstDescProto*>> mem_block_id2regst_descs;
  for (const auto& task : plan.task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      int32_t mem_block_id = pair.second.mem_block_id();
      if (mem_block_id > start_mem_block_id_ && pair.second.consumer_task_id_size() > 0) {
        CHECK(pair.second.enable_reuse_mem());
        mem_block_id2regst_descs[mem_block_id].push_back(&pair.second);
      }
    }
  }
  for (auto& pair : mem_block_id2regst_descs) {
    std::sort(pair.second.begin(), pair.second.end(),
              [&](const RegstDescProto* lhs, const RegstDescProto* rhs) {
                int64_t lhs_order_in_graph = OrderInGraph4TaskId(lhs->producer_task_id());
                int64_t rhs_order_in_graph = OrderInGraph4TaskId(rhs->producer_task_id());
                if (lhs_order_in_graph == rhs_order_in_graph) {
                  CHECK_NE(lhs->mem_block_offset(), rhs->mem_block_offset());
                  return lhs->mem_block_offset() < rhs->mem_block_offset();
                }
                CHECK_NE(lhs_order_in_graph, rhs_order_in_graph);
                return lhs_order_in_graph < rhs_order_in_graph;
              });
    Handler(pair.second);
  }
}

void Improver::Init(const AvailableMemDesc& amd, const Plan& naive_plan) {
  start_mem_block_id_ = Global<IDMgr>::Get()->NewMemBlockId();
  amd_ = amd;
}

Maybe<Plan> Improver::GenAndInferMemBlockIdOnly(const AvailableMemDesc& amd,
                                                const Plan& naive_plan) {
  Init(amd, naive_plan);
  Plan complete_plan = GenAndInferMemBlockId(naive_plan);
  // Check if there is any zone out of memory even though all register_num == 1
  MemZoneRegstDescs mz_regst_descs;
  MakeMemZoneRegstDescs(complete_plan, &mz_regst_descs);
  HashMap<int64_t, double> zero2one{{0, 1}};
  auto Zero2One = [&](int64_t) -> const HashMap<int64_t, double>& { return zero2one; };
  JUST(CheckAllZoneNotOOM(mz_regst_descs, Zero2One, Zero2One, 1));
  SetUniqueMemBlockId4UnreusedMemRegst(&complete_plan);
  GenMemBlockAndChunk4Plan(&complete_plan);
  return complete_plan;
}

Maybe<Plan> Improver::Improve(const AvailableMemDesc& amd, const Plan& naive_plan,
                              const std::string& act_event_filepath) {
  Init(amd, naive_plan);
  std::list<std::unique_ptr<ActEvent>> act_events;
  ParseActEvents(act_event_filepath, &act_events);
  ChainActGraph chain_act_graph(naive_plan, std::move(act_events));

  auto PathDurations4RegstDescId = MakeGetterPathDurations4RegstDescId(chain_act_graph);
  auto PathIIScales4RegstDescId = MakeGetterPathIIScales4RegstDescId(chain_act_graph);
  double base_ii = chain_act_graph.CalcBaseII();

  Plan mem_unlimited_plan(naive_plan);
  ForEachImprovedRegstNum(naive_plan, false, base_ii, PathDurations4RegstDescId,
                          PathIIScales4RegstDescId, MakeSetterSetPlanRegstNum(&mem_unlimited_plan));
  Plan complete_plan = GenAndInferMemBlockId(mem_unlimited_plan);
  Plan plan(complete_plan);
  JUST(ForEachImprovedRegstNum(complete_plan, true, base_ii, PathDurations4RegstDescId,
                               PathIIScales4RegstDescId, MakeSetterSetPlanRegstNum(&plan)));
  FixReliantCtrlRegstNum(plan, MakeGetterGetPlanRegstNum(&plan), MakeSetterSetPlanRegstNum(&plan));
  SetUniqueMemBlockId4UnreusedMemRegst(&plan);
  GenMemBlockAndChunk4Plan(&plan);
  return plan;
}

Plan Improver::GenAndInferMemBlockId(const Plan& naive_plan) const {
  Plan plan(naive_plan);
  PlanTaskGraph plan_task_graph(naive_plan);
  {
    const auto& MutRegstDesc4Id = PlanUtil::MakeMutRegstDesc4Id(&plan);
    if (GlobalJobDesc().use_memory_allocation_algorithm_v2()) {
      IntraJobMemSharingUtil::InferMemBlockId4MemReusedRegst(&plan, plan_task_graph);
    } else {
      ForEachInferredMemBlockId(plan_task_graph, [&](int64_t regst_desc_id, int64_t mem_block_id) {
        MutRegstDesc4Id(regst_desc_id)->set_mem_block_id(mem_block_id);
        MutRegstDesc4Id(regst_desc_id)->set_mem_block_offset(0);
      });
    }
    SetInplaceConsumedRegstDescId(&plan, MutRegstDesc4Id);
  }
  {
    auto OrderInGraph4TaskId = [&](int64_t task_id) {
      return plan_task_graph.TaskProto4TaskId(task_id)->task_set_info().order_in_graph();
    };
    auto IsReachable = [&](int64_t src_task_id, int64_t dst_task_id) {
      return plan_task_graph.IsReachable(src_task_id, dst_task_id);
    };
    ForEachInferredMemBlockCriticalSection(plan, OrderInGraph4TaskId,
                                           MakeSetterAddCtrlRegst(&plan, IsReachable));
  }
  return plan;
}

}  // namespace oneflow
