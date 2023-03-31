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
#include "oneflow/core/job/intra_job_mem_sharing_util.h"
#include <vector>
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/memory_share_strategy.h"
#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/job/plan_util.h"

namespace oneflow {

enum MemAllocAlgoType {
  kMemSizeFirstAlgo = 0,
  kLifetimeFirstAlgo = 1,
  kTimeLineAlgo = 2,
  kMemVolumeFirstAlgo = 3,
};

}  // namespace oneflow

namespace std {

template<>
struct hash<::oneflow::MemAllocAlgoType> {
  std::size_t operator()(const ::oneflow::MemAllocAlgoType& type) const {
    return std::hash<int>()(static_cast<size_t>(type));
  }
};

}  // namespace std

namespace oneflow {

namespace {

int64_t GenDeviceUniqueId(int64_t machine_id, int64_t device_id) {
  return (machine_id << 32) | device_id;
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

struct MemoryChain {
  std::vector<TaskProto*> sorted_tasks;
  HashSet<RegstDescProto*> mem_reused_regsts;
  int64_t total_mem_reused_size = 0;
  Shape time_shape;
};

void InitMemoryChains(Plan* plan,
                      HashMap<int64_t, HashMap<int64_t, MemoryChain>>* device2chain2mem_chain,
                      HashMap<RegstDescProto*, size_t>* mem_reused_regst2size) {
  for (int64_t i = 0; i < plan->task_size(); ++i) {
    TaskProto* task = plan->mutable_task(i);
    const StreamId stream_id = PlanUtil::GetStreamId(*task);
    int64_t machine_id = task->machine_id();
    DeviceType device_type = stream_id.device_id().device_type();
    // TODO(zwx): eliminate this special 'is cpu' determine
    if (device_type == DeviceType::kCPU) { continue; }
    if (!IsValidChainId(task->task_set_info().chain_id())) { continue; }
    int64_t device_id = stream_id.device_id().device_index();
    int64_t device_unique_id = GenDeviceUniqueId(machine_id, device_id);
    MemoryChain* mem_chain =
        &((*device2chain2mem_chain)[device_unique_id][task->task_set_info().chain_id()]);
    mem_chain->sorted_tasks.emplace_back(task);
    for (auto& pair : *(task->mutable_produced_regst_desc())) {
      RegstDescProto* regst_desc = &pair.second;
      int64_t regst_total_main_size = RtRegstDesc(*regst_desc).TotalMainByteSize4AllRegst();
      if (regst_desc->mem_case().device_type() == device_type
          && regst_desc->mem_case().device_id() == device_id && regst_desc->enable_reuse_mem()
          && regst_desc->register_num() == 1 && regst_desc->mem_block_id() == -1
          && regst_desc->mem_block_offset() == -1
          && regst_desc->regst_desc_type().has_data_regst_desc() && regst_total_main_size > 0) {
        CHECK(mem_chain->mem_reused_regsts.insert(regst_desc).second);
        (*mem_reused_regst2size)[regst_desc] = regst_total_main_size;
        mem_chain->total_mem_reused_size += regst_total_main_size;

        // for time shape in mem chain
        Shape regst_time_shape =
            Shape(regst_desc->regst_desc_type().data_regst_desc().time_shape());
        if (!mem_chain->time_shape.is_initialized()) {
          mem_chain->time_shape = regst_time_shape;
        } else {
          CHECK(mem_chain->time_shape == regst_time_shape);
        }
      }
    }
  }
  for (auto& device_pair : *device2chain2mem_chain) {
    HashMap<int64_t, MemoryChain>* chain2mem_chain = &device_pair.second;
    HashSet<int64_t> useless_chain_ids;
    for (auto& pair : *chain2mem_chain) {
      if (pair.second.mem_reused_regsts.empty()) { useless_chain_ids.insert(pair.first); }
    }
    for (int64_t chain_id : useless_chain_ids) { chain2mem_chain->erase(chain_id); }
    for (auto& pair : *chain2mem_chain) {
      MemoryChain* mem_chain = &pair.second;
      std::sort(mem_chain->sorted_tasks.begin(), mem_chain->sorted_tasks.end(),
                [&](const TaskProto* lhs, const TaskProto* rhs) {
                  int64_t lhs_order_in_graph = lhs->task_set_info().order_in_graph();
                  int64_t rhs_order_in_graph = rhs->task_set_info().order_in_graph();
                  CHECK_NE(lhs_order_in_graph, rhs_order_in_graph);
                  return lhs_order_in_graph < rhs_order_in_graph;
                });
    }
  }
}

bool TryMergeMemChain2MergedChains(
    std::vector<MemoryChain*>* merged_chains, MemoryChain* mem_chain,
    const std::function<bool(const MemoryChain*, const MemoryChain*)>& IsStrictOrderL2R) {
  Shape meta_shape({1, 1});
  std::sort(merged_chains->begin(), merged_chains->end(), [&](MemoryChain* lhs, MemoryChain* rhs) {
    return lhs->total_mem_reused_size > rhs->total_mem_reused_size;
  });
  for (MemoryChain* merged_chain : *merged_chains) {
    if (merged_chain->time_shape == meta_shape && mem_chain->time_shape == meta_shape) {
      if (IsStrictOrderL2R(merged_chain, mem_chain)) {
        merged_chain->sorted_tasks.insert(merged_chain->sorted_tasks.end(),
                                          mem_chain->sorted_tasks.begin(),
                                          mem_chain->sorted_tasks.end());
        merged_chain->mem_reused_regsts.insert(mem_chain->mem_reused_regsts.begin(),
                                               mem_chain->mem_reused_regsts.end());
        merged_chain->total_mem_reused_size += mem_chain->total_mem_reused_size;
        return true;
      }
    }
  }
  return false;
}

bool IsReachableToAnyOtherTask(const TaskProto* src_task, const HashSet<int64_t>& task_ids) {
  for (const auto& pair : src_task->produced_regst_desc()) {
    for (int64_t consumer : pair.second.consumer_task_id()) {
      if (task_ids.find(consumer) != task_ids.end()) { return true; }
    }
  }
  return false;
}

bool IsTaskConnectedL2R(const TaskProto* src, const TaskProto* dst) {
  for (const auto& pair : src->produced_regst_desc()) {
    for (int64_t consumer : pair.second.consumer_task_id()) {
      if (consumer == dst->task_id()) { return true; }
    }
  }
  return false;
}

void GenMemChainTasksAndRegsts(
    Plan* plan,
    const std::function<bool(const std::string&, const std::string&)>& IsOpNameDataOrCtrlReachable,
    HashMap<int64_t, std::vector<TaskProto*>>* mem_chain2sorted_tasks,
    HashMap<int64_t, std::vector<RegstDescProto*>>* mem_chain2mem_reused_regsts,
    HashMap<int64_t, HashMap<int64_t, RegstDescProto*>>* mem_chain2regst_desc_id2reuse_regst_desc,
    HashMap<RegstDescProto*, size_t>* mem_reused_regst2size) {
  mem_chain2sorted_tasks->clear();
  mem_chain2mem_reused_regsts->clear();
  HashMap<int64_t, HashMap<int64_t, MemoryChain>> device2chain2mem_chain;
  InitMemoryChains(plan, &device2chain2mem_chain, mem_reused_regst2size);

  auto TryGetTaskNodeLogicalOpName = [&](const TaskProto* task_proto,
                                         std::string* op_name) -> bool {
    if (task_proto->task_type() == TaskType::kNormalForward
        && task_proto->exec_sequence().exec_node_size() == 1) {
      *op_name = PlanUtil::GetOpAttribute(plan, task_proto->job_id(),
                                          task_proto->exec_sequence().exec_node(0).kernel_conf())
                     .op_conf()
                     .name();
      return true;
    }
    return false;
  };

  auto IsStrictOrderL2R = [&](const MemoryChain* lhs, const MemoryChain* rhs) -> bool {
    const TaskProto* l_chain_sink_task_node = lhs->sorted_tasks.back();
    const TaskProto* r_chain_source_task_node = rhs->sorted_tasks.front();
    std::string l_op_name;
    std::string r_op_name;
    if (TryGetTaskNodeLogicalOpName(l_chain_sink_task_node, &l_op_name)
        && TryGetTaskNodeLogicalOpName(r_chain_source_task_node, &r_op_name)) {
      return IsOpNameDataOrCtrlReachable(l_op_name, r_op_name);
    }
    return false;
  };

  int64_t mem_chain_id = 0;

  bool enable_mem_chain_merge =
      Singleton<ResourceDesc, ForSession>::Get()->resource().enable_mem_chain_merge();

  for (auto& device_chain_pair : device2chain2mem_chain) {
    if (device_chain_pair.second.empty()) { continue; }
    // sort
    std::vector<MemoryChain*> mem_chains;
    mem_chains.reserve(device_chain_pair.second.size());
    std::vector<MemoryChain*> merged_chains;
    for (auto& pair : device_chain_pair.second) { mem_chains.emplace_back(&pair.second); }
    std::sort(mem_chains.begin(), mem_chains.end(), [&](MemoryChain* lhs, MemoryChain* rhs) {
      int64_t lhs_order_in_graph = lhs->sorted_tasks.front()->task_set_info().order_in_graph();
      int64_t rhs_order_in_graph = rhs->sorted_tasks.front()->task_set_info().order_in_graph();
      CHECK_NE(lhs_order_in_graph, rhs_order_in_graph);
      return lhs_order_in_graph < rhs_order_in_graph;
    });
    if (enable_mem_chain_merge) {
      for (MemoryChain* mem_chain : mem_chains) {
        if (!TryMergeMemChain2MergedChains(&merged_chains, mem_chain, IsStrictOrderL2R)) {
          merged_chains.emplace_back(mem_chain);
        }
      }
    } else {
      merged_chains.swap(mem_chains);
    }
    for (MemoryChain* merged_chain : merged_chains) {
      std::vector<TaskProto*>* sorted_tasks = &((*mem_chain2sorted_tasks)[mem_chain_id]);
      CHECK(sorted_tasks->empty());
      sorted_tasks->insert(sorted_tasks->end(), merged_chain->sorted_tasks.begin(),
                           merged_chain->sorted_tasks.end());
      std::vector<RegstDescProto*>* mem_reused_regsts =
          &((*mem_chain2mem_reused_regsts)[mem_chain_id]);
      CHECK(mem_reused_regsts->empty());
      mem_reused_regsts->insert(mem_reused_regsts->end(), merged_chain->mem_reused_regsts.begin(),
                                merged_chain->mem_reused_regsts.end());
      // Merge HashSet mem_chain2mem_reused_regsts and HashMap regst_desc_id2reuse_regst_desc
      auto& regst_desc_id2reuse_regst_desc =
          (*mem_chain2regst_desc_id2reuse_regst_desc)[mem_chain_id];
      CHECK(regst_desc_id2reuse_regst_desc.empty());
      for (auto& mem_reused_regst : merged_chain->mem_reused_regsts) {
        regst_desc_id2reuse_regst_desc[mem_reused_regst->regst_desc_id()] = mem_reused_regst;
      }
      ++mem_chain_id;
    }
  }

  CHECK_EQ(mem_chain2sorted_tasks->size(), mem_chain2mem_reused_regsts->size());

  // NOTE(chengcheng): add ctrl safe guard for each mem chain
  HashMap<int64_t, TaskProto*> task_id2proto;
  for (int64_t i = 0; i < plan->task_size(); ++i) {
    TaskProto* task = plan->mutable_task(i);
    CHECK(task_id2proto.emplace(task->task_id(), task).second);
  }
  for (auto& pair : *mem_chain2sorted_tasks) {
    std::vector<TaskProto*>* sorted_tasks = &(pair.second);
    // NOTE(chengcheng): We CANNOT only add ctrl safe guard between first and last task,
    //  because of the sorted_tasks may connected as a graph, has multi-tail tasks(sink task).
    const std::vector<RegstDescProto*>& mem_reused_regsts =
        mem_chain2mem_reused_regsts->at(pair.first);
    if (mem_reused_regsts.size() <= 1) { continue; }

    HashSet<int64_t> consumer_task_ids;
    for (const RegstDescProto* regst : mem_reused_regsts) {
      for (int64_t consumer : regst->consumer_task_id()) { consumer_task_ids.insert(consumer); }
    }
    std::vector<TaskProto*> sink_tasks;
    sink_tasks.reserve(consumer_task_ids.size());
    for (int64_t src_task_id : consumer_task_ids) {
      auto it = task_id2proto.find(src_task_id);
      CHECK(it != task_id2proto.end());
      if (!IsReachableToAnyOtherTask(it->second, consumer_task_ids)) {
        sink_tasks.emplace_back(it->second);
      }
    }

    TaskProto* first_task = sorted_tasks->front();
    for (TaskProto* sink_task : sink_tasks) {
      CHECK(first_task != sink_task);
      if (!IsTaskConnectedL2R(first_task, sink_task)) {
        TryConnectWithMemSafeGuardCtrlRegstDesc(first_task, sink_task);
      }
    }
  }
}

void GenRegstAllocFreeTimeLineAndRegstLifetimes(
    const std::vector<TaskProto*>& sorted_tasks,
    const std::vector<RegstDescProto*>& mem_reused_regsts,
    const HashMap<int64_t, RegstDescProto*>& regst_desc_id2reuse_regst_desc,
    const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
    HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>* regst2lifetime,
    HashMap<RegstDescProto*, RegstDescProto*>* consumer2inplaced_regst, size_t* peak_memory) {
  CHECK(consumer2inplaced_regst->empty());
  std::vector<std::vector<RegstDescProto*>> alloc_regsts_timeline(sorted_tasks.size());
  std::vector<std::vector<RegstDescProto*>> free_regsts_timeline(sorted_tasks.size());
  HashMap<int64_t, int64_t> task_id2sorted_id;
  for (int64_t i = 0; i < sorted_tasks.size(); ++i) {
    TaskProto* task = sorted_tasks.at(i);
    CHECK(task_id2sorted_id.emplace(task->task_id(), i).second);
  }

  auto FindLastFreeIndexInSortedTasks = [&](RegstDescProto* regst_desc) -> int64_t {
    // temp regst will set free index as same as alloc index
    int64_t free_index = task_id2sorted_id.at(regst_desc->producer_task_id());
    for (int64_t consumer_task_id : regst_desc->consumer_task_id()) {
      // if consumer is not in this mem chain, set free index = last index
      int64_t this_sorted_index = sorted_tasks.size() - 1;
      if (task_id2sorted_id.find(consumer_task_id) != task_id2sorted_id.end()) {
        this_sorted_index = task_id2sorted_id.at(consumer_task_id);
      }
      free_index = std::max(free_index, this_sorted_index);
    }
    return free_index;
  };

  auto TryFindFirstInplacedRegstDesc = [&](RegstDescProto* consumer_regst) -> RegstDescProto* {
    RegstDescProto* inplaced_regst = nullptr;
    while (consumer_regst->has_hint_inplace_consumed_regst_desc_id()
           && consumer_regst->hint_inplace_consumed_regst_desc_id() != -1) {
      const auto& iterator_hint_inplaced_regst = regst_desc_id2reuse_regst_desc.find(
          consumer_regst->hint_inplace_consumed_regst_desc_id());
      if (iterator_hint_inplaced_regst != regst_desc_id2reuse_regst_desc.end()) {
        inplaced_regst = iterator_hint_inplaced_regst->second;
        consumer_regst = iterator_hint_inplaced_regst->second;
      } else {
        break;
      }
    }
    return inplaced_regst;
  };

  HashMap<int64_t, int64_t> regst_desc_id2free_index;
  for (RegstDescProto* regst_desc : mem_reused_regsts) {
    RegstDescProto* inplaced_regst_desc = TryFindFirstInplacedRegstDesc(regst_desc);
    if (inplaced_regst_desc != nullptr) {
      CHECK(consumer2inplaced_regst->emplace(regst_desc, inplaced_regst_desc).second);
      continue;
    }

    alloc_regsts_timeline[task_id2sorted_id.at(regst_desc->producer_task_id())].push_back(
        regst_desc);
    CHECK(regst_desc_id2free_index
              .emplace(regst_desc->regst_desc_id(), FindLastFreeIndexInSortedTasks(regst_desc))
              .second);
  }
  // inplace extend regst free index
  for (auto pair : *consumer2inplaced_regst) {
    RegstDescProto* consumer_regst_desc = pair.first;
    int64_t inplaced_regst_desc_id = pair.second->regst_desc_id();
    CHECK(regst_desc_id2free_index.find(inplaced_regst_desc_id) != regst_desc_id2free_index.end());
    regst_desc_id2free_index.at(inplaced_regst_desc_id) =
        std::max(regst_desc_id2free_index.at(inplaced_regst_desc_id),
                 FindLastFreeIndexInSortedTasks(consumer_regst_desc));
  }
  for (const auto& pair : regst_desc_id2free_index) {
    free_regsts_timeline[pair.second].push_back(regst_desc_id2reuse_regst_desc.at(pair.first));
  }

  HashSet<RegstDescProto*> remain_regsts;
  size_t remain_memory = 0;
  *peak_memory = 0;
  for (int64_t i = 0; i < sorted_tasks.size(); ++i) {
    for (RegstDescProto* alloc_regst : alloc_regsts_timeline.at(i)) {
      // Record the born time
      (*regst2lifetime)[alloc_regst].first = i;
      CHECK(remain_regsts.insert(alloc_regst).second);
      remain_memory += mem_reused_regst2size.at(alloc_regst);
      // NOTE(chengcheng): insert time line to regst proto
      alloc_regst->set_mem_block_total_actor_count(sorted_tasks.size());
      alloc_regst->set_alloc_before_actor(i);
    }
    // Update the peak of memory during execution
    if (*peak_memory < remain_memory) { *peak_memory = remain_memory; }
    for (RegstDescProto* free_regst : free_regsts_timeline.at(i)) {
      CHECK_EQ(remain_regsts.erase(free_regst), 1);
      free_regst->set_free_after_actor(i);
      remain_memory -= mem_reused_regst2size.at(free_regst);
      // Record the die time
      (*regst2lifetime)[free_regst].second = i + 1;
    }
  }
  // Make sure that every register has a die time
  CHECK(remain_regsts.empty());
}

void MemReusedLifetimeFirstAlgo(
    const bool compact_insert,
    const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& regst2lifetime,
    const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
    MemBlockResultInfo<RegstDescProto*>* result) {
  std::vector<RegstDescProto*> order;
  order.reserve(regst2lifetime.size());
  for (const auto& pair : regst2lifetime) { order.emplace_back(pair.first); }
  std::sort(order.begin(), order.end(), [&](RegstDescProto* lhs, RegstDescProto* rhs) {
    int64_t l_value = regst2lifetime.at(lhs).second - regst2lifetime.at(lhs).first;
    int64_t r_value = regst2lifetime.at(rhs).second - regst2lifetime.at(rhs).first;
    if (l_value == r_value) { return regst2lifetime.at(lhs).first < regst2lifetime.at(rhs).first; }
    return l_value > r_value;
  });
  MemReusedAlgorithmAllocateByOrder(compact_insert, order, mem_reused_regst2size, regst2lifetime,
                                    result);
}

void MemReusedTimeLineAlgo(
    const bool compact_insert,
    const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& regst2lifetime,
    const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
    MemBlockResultInfo<RegstDescProto*>* result) {
  std::vector<RegstDescProto*> order;
  order.reserve(regst2lifetime.size());
  for (const auto& pair : regst2lifetime) { order.emplace_back(pair.first); }
  std::sort(order.begin(), order.end(), [&](RegstDescProto* lhs, RegstDescProto* rhs) {
    int64_t l_value = regst2lifetime.at(lhs).first;
    int64_t r_value = regst2lifetime.at(rhs).first;
    if (l_value == r_value) {
      return regst2lifetime.at(lhs).second > regst2lifetime.at(rhs).second;
    }
    return l_value > r_value;
  });
  MemReusedAlgorithmAllocateByOrder(compact_insert, order, mem_reused_regst2size, regst2lifetime,
                                    result);
}

void MemReusedMemVolumeFirstAlgo(
    const bool compact_insert,
    const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& regst2lifetime,
    const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
    MemBlockResultInfo<RegstDescProto*>* result) {
  std::vector<RegstDescProto*> order;
  order.reserve(regst2lifetime.size());
  auto ComputeMemoryVolume = [&](RegstDescProto* key) {
    return mem_reused_regst2size.at(key)
           * (regst2lifetime.at(key).second - regst2lifetime.at(key).first) / 1000;
  };
  for (const auto& pair : regst2lifetime) { order.emplace_back(pair.first); }
  std::sort(order.begin(), order.end(), [&](RegstDescProto* lhs, RegstDescProto* rhs) {
    size_t l_value = ComputeMemoryVolume(lhs);
    size_t r_value = ComputeMemoryVolume(rhs);
    if (l_value == r_value) {
      return mem_reused_regst2size.at(lhs) > mem_reused_regst2size.at(rhs);
    }
    return l_value > r_value;
  });
  MemReusedAlgorithmAllocateByOrder(compact_insert, order, mem_reused_regst2size, regst2lifetime,
                                    result);
}

void SelectAlgorithmGenMemBlockOffset4Regsts(
    MemAllocAlgoType algo_id, const bool compact_insert,
    const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& regst2lifetime,
    const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
    MemBlockResultInfo<RegstDescProto*>* result) {
  CHECK_EQ(result->mem_block_size, 0);
  CHECK(result->regst_desc2offset.empty());

  switch (algo_id) {
    case kMemSizeFirstAlgo:
      MemReusedMemSizeFirstAlgo(compact_insert, regst2lifetime, mem_reused_regst2size, result);
      break;
    case kLifetimeFirstAlgo:
      MemReusedLifetimeFirstAlgo(compact_insert, regst2lifetime, mem_reused_regst2size, result);
      break;
    case kTimeLineAlgo:
      MemReusedTimeLineAlgo(compact_insert, regst2lifetime, mem_reused_regst2size, result);
      break;
    case kMemVolumeFirstAlgo:
      MemReusedMemVolumeFirstAlgo(compact_insert, regst2lifetime, mem_reused_regst2size, result);
      break;
    default: UNIMPLEMENTED();
  }
  CHECK_GT(result->mem_block_size, 0);
  CHECK(!result->regst_desc2offset.empty());
}

int64_t CountMemAllocAlgoNum() {
  const MemoryAllocationAlgorithmConf& mem_alloc_algo_conf =
      GlobalJobDesc().job_conf().memory_allocation_algorithm_conf();
  int64_t alloc_algo_num = 0;
  if (mem_alloc_algo_conf.use_mem_size_first_algo()) { ++alloc_algo_num; }
  if (mem_alloc_algo_conf.use_lifetime_first_algo()) { ++alloc_algo_num; }
  if (mem_alloc_algo_conf.use_time_line_algo()) { ++alloc_algo_num; }
  if (mem_alloc_algo_conf.use_mem_volume_first_algo()) { ++alloc_algo_num; }
  CHECK_GE(alloc_algo_num, 0) << "At least choose one type of memory allocation algorithm. We "
                                 "recommend use_mem_size_first_algo()";
  const MemoryCompactInsertConf& mem_compact_insert_conf =
      GlobalJobDesc().job_conf().memory_compact_insert_conf();
  int64_t compact_insert_num = 0;
  if (mem_compact_insert_conf.use_compact_insert()) { ++compact_insert_num; }
  if (mem_compact_insert_conf.use_non_compact_insert()) { ++compact_insert_num; }
  CHECK_GE(compact_insert_num, 0) << "At least choose one type of memory arrangement algorithm "
                                     "during memory allocation. We recommend use_compact_insert()";

  return alloc_algo_num * compact_insert_num;
}

void InitAlgo2Result(
    HashMap<std::pair<MemAllocAlgoType, bool>, MemBlockResultInfo<RegstDescProto*>>* algo2result) {
  CHECK(algo2result->empty());
  std::vector<bool> compact_insert_algorithms;
  const MemoryCompactInsertConf& mem_compact_insert_conf =
      GlobalJobDesc().job_conf().memory_compact_insert_conf();
  if (mem_compact_insert_conf.use_compact_insert()) { compact_insert_algorithms.push_back(true); }
  if (mem_compact_insert_conf.use_non_compact_insert()) {
    compact_insert_algorithms.push_back(false);
  }

  const MemoryAllocationAlgorithmConf& mem_alloc_algo_conf =
      GlobalJobDesc().job_conf().memory_allocation_algorithm_conf();
  // NOTE: Experiments show that memory first might be good enough for some cases.
  for (auto compact_insert : compact_insert_algorithms) {
    if (mem_alloc_algo_conf.use_mem_size_first_algo()) {
      (*algo2result)[{kMemSizeFirstAlgo, compact_insert}] = MemBlockResultInfo<RegstDescProto*>();
    }
    if (mem_alloc_algo_conf.use_lifetime_first_algo()) {
      (*algo2result)[{kLifetimeFirstAlgo, compact_insert}] = MemBlockResultInfo<RegstDescProto*>();
    }
    if (mem_alloc_algo_conf.use_time_line_algo()) {
      (*algo2result)[{kTimeLineAlgo, compact_insert}] = MemBlockResultInfo<RegstDescProto*>();
    }
    if (mem_alloc_algo_conf.use_mem_volume_first_algo()) {
      (*algo2result)[{kMemVolumeFirstAlgo, compact_insert}] = MemBlockResultInfo<RegstDescProto*>();
    }
  }
}

}  // namespace

void IntraJobMemSharingUtil::InferMemBlockId4MemReusedRegst(
    Plan* plan, const std::function<bool(const std::string&, const std::string&)>&
                    IsOpNameDataOrCtrlReachable) {
  // 1 device 1 mem chain
  HashMap<int64_t, std::vector<TaskProto*>> mem_chain2sorted_tasks;
  HashMap<int64_t, std::vector<RegstDescProto*>> mem_chain2mem_reused_regsts;
  // NOTE: We only store those reusable registers in mem_chain2regst_desc_id2reuse_regst_desc.
  //      There are no duplicated registers in different memory chains.
  HashMap<int64_t, HashMap<int64_t, RegstDescProto*>> mem_chain2regst_desc_id2reuse_regst_desc;
  HashMap<RegstDescProto*, size_t> mem_reused_regst2size;
  GenMemChainTasksAndRegsts(plan, IsOpNameDataOrCtrlReachable, &mem_chain2sorted_tasks,
                            &mem_chain2mem_reused_regsts, &mem_chain2regst_desc_id2reuse_regst_desc,
                            &mem_reused_regst2size);
  if (mem_chain2mem_reused_regsts.empty()) { return; }
  HashSet<int64_t> mem_chains;
  for (const auto& pair : mem_chain2mem_reused_regsts) { mem_chains.insert(pair.first); }
  // register lifetime
  HashMap<int64_t, HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>> mem_chain2regst2lifetime;
  // info for inplace
  HashMap<int64_t, HashMap<RegstDescProto*, RegstDescProto*>> mem_chain2consumer2inplaced_regst;
  // info for straighten
  HashMap<int64_t, size_t> mem_chain2peak_memory;

  // step 1: generate regst alloc/free queue AND regst lifetimes
  for (const auto& pair : mem_chain2mem_reused_regsts) {
    GenRegstAllocFreeTimeLineAndRegstLifetimes(
        mem_chain2sorted_tasks.at(pair.first), pair.second,
        mem_chain2regst_desc_id2reuse_regst_desc.at(pair.first), mem_reused_regst2size,
        &mem_chain2regst2lifetime[pair.first], &mem_chain2consumer2inplaced_regst[pair.first],
        &mem_chain2peak_memory[pair.first]);
  }

  // step 2: multi-thread run several algorithm for each mem chain
  HashMap<int64_t, HashMap<std::pair<MemAllocAlgoType, bool>, MemBlockResultInfo<RegstDescProto*>>>
      mem_chain2algo2result;
  {
    int64_t work_size = mem_chain2mem_reused_regsts.size() * CountMemAllocAlgoNum();
    int64_t thread_pool_size = std::min<int64_t>(work_size, std::thread::hardware_concurrency());
    BlockingCounter counter(work_size);
    ThreadPool thread_pool(thread_pool_size);
    for (int64_t mem_chain_id : mem_chains) {
      InitAlgo2Result(&mem_chain2algo2result[mem_chain_id]);
      for (auto& pair : mem_chain2algo2result.at(mem_chain_id)) {
        MemAllocAlgoType algo_id = pair.first.first;
        bool compact_insert = pair.first.second;
        MemBlockResultInfo<RegstDescProto*>* result = &pair.second;
        thread_pool.AddWork([algo_id, compact_insert, mem_chain_id, &mem_chain2regst2lifetime,
                             &mem_reused_regst2size, result, &counter]() {
          SelectAlgorithmGenMemBlockOffset4Regsts(algo_id, compact_insert,
                                                  mem_chain2regst2lifetime.at(mem_chain_id),
                                                  mem_reused_regst2size, result);
          counter.Decrease();
        });
      }
    }
    counter.WaitForeverUntilCntEqualZero();
  }

  // step 3: choose best one for each mem chain and set offset for inplace consumer regst
  for (auto& pair : mem_chain2algo2result) {
    MemBlockResultInfo<RegstDescProto*>* best_result = nullptr;
    for (auto& algo_result_pair : pair.second) {
      if (!best_result || algo_result_pair.second.mem_block_size < best_result->mem_block_size) {
        best_result = &algo_result_pair.second;
      }
    }
    CHECK(best_result != nullptr);

    // Update the offset with a smaller total memory size if the current size is greater than the
    // lower bound
    if (GlobalJobDesc().job_conf().enable_compress_memory()) {
      MemoryShareStrategy mss;
      mss.AdaptivelyUpdateOffset(mem_reused_regst2size, mem_chain2regst2lifetime.at(pair.first),
                                 mem_chain2peak_memory[pair.first], &best_result->mem_block_size,
                                 &best_result->regst_desc2offset);
    }

    int64_t mem_block_id = Singleton<IDMgr>::Get()->NewMemBlockId();
    CHECK_EQ(mem_chain2mem_reused_regsts.at(pair.first).size(),
             (best_result->regst_desc2offset.size()
              + mem_chain2consumer2inplaced_regst.at(pair.first).size()));
    for (const auto& regst_offset_pair : best_result->regst_desc2offset) {
      RegstDescProto* regst_desc = regst_offset_pair.first;
      CHECK_EQ(regst_desc->mem_block_id(), -1);
      regst_desc->set_mem_block_id(mem_block_id);
      regst_desc->set_mem_block_offset(regst_offset_pair.second);
    }
    // set inplace
    for (auto& consumer_inplace_pair : mem_chain2consumer2inplaced_regst.at(pair.first)) {
      RegstDescProto* consumer_regst_desc = consumer_inplace_pair.first;
      CHECK_EQ(consumer_regst_desc->mem_block_id(), -1);
      RegstDescProto* inplaced_regst_desc = consumer_inplace_pair.second;
      CHECK_EQ(inplaced_regst_desc->mem_block_id(), mem_block_id);
      CHECK_NE(inplaced_regst_desc->mem_block_offset(), -1);
      consumer_regst_desc->set_mem_block_id(inplaced_regst_desc->mem_block_id());
      consumer_regst_desc->set_mem_block_offset(inplaced_regst_desc->mem_block_offset());
    }

    // set inplace hint and check
    const auto& regst_desc_id2reuse_regst_desc =
        mem_chain2regst_desc_id2reuse_regst_desc.at(pair.first);
    for (auto& consumer_inplace_pair : mem_chain2consumer2inplaced_regst.at(pair.first)) {
      RegstDescProto* consumer_regst_desc = consumer_inplace_pair.first;
      RegstDescProto* inplaced_regst_desc = consumer_inplace_pair.second;
      CHECK(consumer_regst_desc->has_inplace_consumed_regst_desc_id() == false);
      CHECK(consumer_regst_desc->has_hint_inplace_consumed_regst_desc_id());
      int64_t hint = consumer_regst_desc->hint_inplace_consumed_regst_desc_id();
      // NOTE(chengcheng): hint regst desc id may NOT be the inplaced_regst_desc_id
      //   because of nest inplace.
      // NOTE: All the registers in mem_chain2consumer2inplaced_regst are reusable
      auto hint_it = regst_desc_id2reuse_regst_desc.find(hint);
      CHECK(hint_it != regst_desc_id2reuse_regst_desc.end());
      RegstDescProto* in_regst_desc = hint_it->second;
      CHECK_EQ(consumer_regst_desc->mem_block_id(), in_regst_desc->mem_block_id());
      CHECK_EQ(consumer_regst_desc->mem_block_offset(), in_regst_desc->mem_block_offset());
      CHECK_EQ(in_regst_desc->mem_block_offset(), inplaced_regst_desc->mem_block_offset());
      CHECK_EQ(consumer_regst_desc->register_num(), in_regst_desc->register_num());
      consumer_regst_desc->set_inplace_consumed_regst_desc_id(hint);
    }
  }
}

}  // namespace oneflow
