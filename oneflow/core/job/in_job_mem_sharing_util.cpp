#include "oneflow/core/job/in_job_mem_sharing_util.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

namespace {

const size_t kMemReuseAlgorithmNum = 3;

int64_t GenDeviceUniqueId(int64_t machine_id, int64_t device_id) {
  return (machine_id << 32) | device_id;
}

void GenRegstDescId2RegstDesc(Plan* plan,
                              HashMap<int64_t, RegstDescProto*>* regst_desc_id2regst_desc) {
  regst_desc_id2regst_desc->clear();
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      int64_t regst_desc_id = pair.second.regst_desc_id();
      regst_desc_id2regst_desc->insert({regst_desc_id, &pair.second});
    }
  }
}

void GenMemChainTasksAndRegsts(
    Plan* plan, HashMap<int64_t, std::vector<TaskProto*>>* device_unique_id2tasks,
    HashMap<int64_t, HashSet<RegstDescProto*>>* device_unique_id2regsts) {
  Shape meta_shape({GlobalJobDesc().TotalBatchNum(), GlobalJobDesc().NumOfPiecesInBatch()});
  device_unique_id2tasks->clear();
  device_unique_id2regsts->clear();
  for (int64_t i = 0; i < plan->task_size(); ++i) {
    TaskProto* task = plan->mutable_task(i);
    int64_t machine_id = task->machine_id();
    DeviceType device_type = Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(task->thrd_id());
    if (device_type == DeviceType::kCPU) { continue; }
    int64_t device_id = Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(task->thrd_id());
    int64_t device_unique_id = GenDeviceUniqueId(machine_id, device_id);
    (*device_unique_id2tasks)[device_unique_id].push_back(task);
    for (auto& pair : *(task->mutable_produced_regst_desc())) {
      RegstDescProto* regst_desc = &pair.second;
      if (regst_desc->mem_case().has_device_cuda_mem()
          && regst_desc->mem_case().device_cuda_mem().device_id() == device_id
          && regst_desc->enable_reuse_mem() && regst_desc->register_num() == 1
          && regst_desc->mem_block_id() == -1 && regst_desc->mem_block_offset() == -1
          && regst_desc->regst_desc_type().has_data_regst_desc()
          && Shape(regst_desc->regst_desc_type().data_regst_desc().time_shape()) == meta_shape) {
        CHECK((*device_unique_id2regsts)[device_unique_id].insert(regst_desc).second);
      }
    }
  }
  for (auto& pair : *device_unique_id2tasks) {
    std::sort(pair.second.begin(), pair.second.end(),
              [&](const TaskProto* lhs, const TaskProto* rhs) {
                int64_t lhs_order_in_graph = lhs->task_set_info().order_in_graph();
                int64_t rhs_order_in_graph = rhs->task_set_info().order_in_graph();
                CHECK_NE(lhs_order_in_graph, rhs_order_in_graph);
                return lhs_order_in_graph < rhs_order_in_graph;
              });
  }
}

void GenRegstApplyReleaseQueueAndRegstMutualExclusions(
    const std::vector<TaskProto*>* sorted_tasks, const HashSet<RegstDescProto*>* mem_reused_regsts,
    const HashMap<int64_t, RegstDescProto*>* regst_desc_id2regst_desc,
    std::vector<HashSet<RegstDescProto*>>* apply_regsts_queue,
    std::vector<HashSet<RegstDescProto*>>* release_regsts_queue,
    HashMap<RegstDescProto*, HashSet<RegstDescProto*>>* regst2mutual_exclusion_regsts,
    HashSet<RegstDescProto*>* inplace_consumer_regst_descs) {
  apply_regsts_queue->clear();
  release_regsts_queue->clear();
  regst2mutual_exclusion_regsts->clear();
  inplace_consumer_regst_descs->clear();
  apply_regsts_queue->resize(sorted_tasks->size());
  release_regsts_queue->resize(sorted_tasks->size());
  HashMap<int64_t, int64_t> task_id2sorted_id;
  for (int64_t i = 0; i < sorted_tasks->size(); ++i) {
    TaskProto* task = sorted_tasks->at(i);
    CHECK(task_id2sorted_id.emplace(task->task_id(), i).second);
  }

  auto FindLastReleaseIndexInSortedTasks = [&](RegstDescProto* regst_desc) -> int64_t {
    // temp regst will set release index as same as apply index
    int64_t release_index = task_id2sorted_id.at(regst_desc->producer_task_id());
    for (int64_t consumer_task_id : regst_desc->consumer_task_id()) {
      // if consumer is not in this mem chain, set release index = last index
      int64_t this_sorted_index = sorted_tasks->size() - 1;
      if (task_id2sorted_id.find(consumer_task_id) != task_id2sorted_id.end()) {
        this_sorted_index = task_id2sorted_id.at(consumer_task_id);
      }
      release_index = std::max(release_index, this_sorted_index);
    }
    return release_index;
  };

  HashMap<int64_t, int64_t> regst_desc_id2release_index;
  for (RegstDescProto* regst_desc : *mem_reused_regsts) {
    if (regst_desc->has_hint_inplace_consumed_regst_desc_id()
        && regst_desc->hint_inplace_consumed_regst_desc_id() != -1) {
      RegstDescProto* inplaced_regst_desc =
          regst_desc_id2regst_desc->at(regst_desc->hint_inplace_consumed_regst_desc_id());
      if (mem_reused_regsts->find(inplaced_regst_desc) != mem_reused_regsts->end()) {
        CHECK(inplace_consumer_regst_descs->insert(regst_desc).second);
        continue;
      }
    }
    CHECK(apply_regsts_queue->at(task_id2sorted_id.at(regst_desc->producer_task_id()))
              .insert(regst_desc)
              .second);
    CHECK(regst_desc_id2release_index
              .emplace(regst_desc->regst_desc_id(), FindLastReleaseIndexInSortedTasks(regst_desc))
              .second);
  }
  // inplace extend regst release index
  for (RegstDescProto* inplace_consumer_regst_desc : *inplace_consumer_regst_descs) {
    int64_t inplaced_regst_desc_id =
        inplace_consumer_regst_desc->hint_inplace_consumed_regst_desc_id();
    CHECK(regst_desc_id2release_index.find(inplaced_regst_desc_id)
          != regst_desc_id2release_index.end());
    regst_desc_id2release_index.at(inplaced_regst_desc_id) =
        std::max(regst_desc_id2release_index.at(inplaced_regst_desc_id),
                 FindLastReleaseIndexInSortedTasks(inplace_consumer_regst_desc));
  }
  for (const auto& pair : regst_desc_id2release_index) {
    CHECK(release_regsts_queue->at(pair.second)
              .insert(regst_desc_id2regst_desc->at(pair.first))
              .second);
  }

  HashSet<RegstDescProto*> remain_regsts;
  for (int64_t i = 0; i < sorted_tasks->size(); ++i) {
    for (RegstDescProto* apply_regst : apply_regsts_queue->at(i)) {
      for (RegstDescProto* remain_regst : remain_regsts) {
        CHECK((*regst2mutual_exclusion_regsts)[apply_regst].insert(remain_regst).second);
        CHECK((*regst2mutual_exclusion_regsts)[remain_regst].insert(apply_regst).second);
      }
      CHECK(remain_regsts.insert(apply_regst).second);
    }
    for (RegstDescProto* release_regst : release_regsts_queue->at(i)) {
      auto it = remain_regsts.find(release_regst);
      CHECK(it != remain_regsts.end());
      remain_regsts.erase(it);
    }
  }
  CHECK(remain_regsts.empty());
}

struct MemBlockResultInfo {
  int64_t mem_block_size;
  HashMap<RegstDescProto*, int64_t> regst_desc2offset;
};

void MemReusedAlgorithm0_OfColorImproved(
    const HashMap<RegstDescProto*, HashSet<RegstDescProto*>>& regst2mutual_exclusion_regsts,
    MemBlockResultInfo* result) {
  TODO();
}

void MemReusedAlgorithm1_TfBfcImproved(
    const std::vector<HashSet<RegstDescProto*>>& apply_regsts_queue,
    const std::vector<HashSet<RegstDescProto*>>& release_regsts_queue, MemBlockResultInfo* result) {
  TODO();
}

void MemReusedAlgorithm2_MinOrderGrowth(
    const HashMap<RegstDescProto*, HashSet<RegstDescProto*>>& regst2mutual_exclusion_regsts,
    MemBlockResultInfo* result) {
  // idea by LiKe
  TODO();
}

void SelectAlgorithmGenMemBlockOffset4Regsts(
    int64_t algo_id, const std::vector<HashSet<RegstDescProto*>>& apply_regsts_queue,
    const std::vector<HashSet<RegstDescProto*>>& release_regsts_queue,
    const HashMap<RegstDescProto*, HashSet<RegstDescProto*>>& regst2mutual_exclusion_regsts,
    MemBlockResultInfo* result) {
  switch (algo_id) {
    case 0: MemReusedAlgorithm0_OfColorImproved(regst2mutual_exclusion_regsts, result); break;
    case 1:
      MemReusedAlgorithm1_TfBfcImproved(apply_regsts_queue, release_regsts_queue, result);
      break;
    case 2: MemReusedAlgorithm2_MinOrderGrowth(regst2mutual_exclusion_regsts, result); break;
    default: UNIMPLEMENTED();
  }
}

}  // namespace

void InJobMemSharingUtil::InferMemBlockId4MemReusedRegst(Plan* plan) {
  // 1 device 1 mem chain
  HashMap<int64_t, std::vector<TaskProto*>> mem_chain2sorted_tasks;
  HashMap<int64_t, HashSet<RegstDescProto*>> mem_chain2mem_reused_regsts;
  GenMemChainTasksAndRegsts(plan, &mem_chain2sorted_tasks, &mem_chain2mem_reused_regsts);
  if (mem_chain2sorted_tasks.size() == 0) { return; }
  HashMap<int64_t, RegstDescProto*> regst_desc_id2regst_desc;
  GenRegstDescId2RegstDesc(plan, &regst_desc_id2regst_desc);
  // info for algorithm
  HashMap<int64_t, std::vector<HashSet<RegstDescProto*>>> mem_chain2task2apply_regsts;
  HashMap<int64_t, std::vector<HashSet<RegstDescProto*>>> mem_chain2task2release_regsts;
  HashMap<int64_t, HashMap<RegstDescProto*, HashSet<RegstDescProto*>>>
      mem_chain2regst2mutual_exclusion_regsts;
  // info for inplace
  HashMap<int64_t, HashSet<RegstDescProto*>> mem_chain2inplace_consumer_regst_descs;

  // step 1: generate regst apply/release queue AND regst mutual exclusions
  for (const auto& pair : mem_chain2sorted_tasks) {
    GenRegstApplyReleaseQueueAndRegstMutualExclusions(
        &pair.second, &mem_chain2mem_reused_regsts.at(pair.first), &regst_desc_id2regst_desc,
        &mem_chain2task2apply_regsts[pair.first], &mem_chain2task2release_regsts[pair.first],
        &mem_chain2regst2mutual_exclusion_regsts[pair.first],
        &mem_chain2inplace_consumer_regst_descs[pair.first]);
  }

  // step 2: multi-thread run several algorithm for each mem chain
  HashMap<int64_t, std::vector<MemBlockResultInfo>> mem_chain2results;
  {
    int64_t cpu_num = std::thread::hardware_concurrency();
    int64_t work_size = mem_chain2sorted_tasks.size() * kMemReuseAlgorithmNum;
    int64_t thread_pool_size = std::min<int64_t>(work_size, cpu_num);
    BlockingCounter counter(work_size);
    ThreadPool thread_pool(thread_pool_size);
    for (const auto& pair : mem_chain2sorted_tasks) {
      int64_t mem_chain_id = pair.first;
      std::vector<MemBlockResultInfo>* results = &mem_chain2results[mem_chain_id];
      results->resize(kMemReuseAlgorithmNum);
      for (int64_t algo_id = 0; algo_id < kMemReuseAlgorithmNum; ++algo_id) {
        MemBlockResultInfo* result = &(results->at(algo_id));
        thread_pool.AddWork([algo_id, mem_chain_id, &mem_chain2task2apply_regsts,
                             &mem_chain2task2release_regsts,
                             &mem_chain2regst2mutual_exclusion_regsts, result, &counter]() {
          SelectAlgorithmGenMemBlockOffset4Regsts(
              algo_id, mem_chain2task2apply_regsts.at(mem_chain_id),
              mem_chain2task2release_regsts.at(mem_chain_id),
              mem_chain2regst2mutual_exclusion_regsts.at(mem_chain_id), result);
          counter.Decrease();
        });
      }
    }
    counter.WaitUntilCntEqualZero();
  }

  // step 3: choose best one for each mem chain and set offset for inplace consumer regst
  for (const auto& pair : mem_chain2results) {
    const std::vector<MemBlockResultInfo>& results = pair.second;
    int64_t best_algo_id = 0;
    for (int64_t algo_id = 0; algo_id < kMemReuseAlgorithmNum; ++algo_id) {
      if (!mem_chain2mem_reused_regsts.at(pair.first).empty()) {
        CHECK_GT(results.at(algo_id).mem_block_size, 0);
        CHECK(!results.at(algo_id).regst_desc2offset.empty());
      }
      if (results.at(algo_id).mem_block_size < results.at(best_algo_id).mem_block_size) {
        best_algo_id = algo_id;
      }
    }
    LOG(INFO) << "mem reuse algorithm choose result : algorithm " << best_algo_id
              << " is best in mem_chain " << pair.first;
    int64_t mem_block_id = Global<IDMgr>::Get()->NewMemBlockId();
    const MemBlockResultInfo& best_result = results.at(best_algo_id);
    for (const auto& regst_offset_pair : best_result.regst_desc2offset) {
      RegstDescProto* regst_desc = regst_offset_pair.first;
      int64_t offset = regst_offset_pair.second;
      CHECK_EQ(regst_desc->mem_block_id(), -1);
      regst_desc->set_mem_block_id(mem_block_id);
      regst_desc->set_mem_block_offset(offset);
    }
    // set inplace
    for (RegstDescProto* inplace_consumer_regst_desc :
         mem_chain2inplace_consumer_regst_descs.at(pair.first)) {
      CHECK_EQ(inplace_consumer_regst_desc->mem_block_id(), -1);
      RegstDescProto* inplaced_regst_desc = regst_desc_id2regst_desc.at(
          inplace_consumer_regst_desc->hint_inplace_consumed_regst_desc_id());
      CHECK_EQ(inplaced_regst_desc->mem_block_id(), mem_block_id);
      CHECK_NE(inplaced_regst_desc->mem_block_offset(), -1);
      inplace_consumer_regst_desc->set_mem_block_id(inplaced_regst_desc->mem_block_id());
      inplace_consumer_regst_desc->set_mem_block_offset(inplaced_regst_desc->mem_block_offset());
    }
  }
}

}  // namespace oneflow
