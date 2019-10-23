#include "oneflow/core/job/intra_job_mem_sharing_util.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

namespace {

const size_t kMemReuseAlgorithmNum = 1;

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

struct MemoryChain {
  std::vector<TaskProto*> sorted_tasks;
  HashSet<RegstDescProto*> mem_reused_regsts;
};

void InitMemoryChains(Plan* plan,
                      HashMap<int64_t, HashMap<int64_t, MemoryChain>>* device2chain2mem_chain) {
  Shape meta_shape({GlobalJobDesc().TotalBatchNum(), GlobalJobDesc().NumOfPiecesInBatch()});
  for (int64_t i = 0; i < plan->task_size(); ++i) {
    TaskProto* task = plan->mutable_task(i);
    int64_t machine_id = task->machine_id();
    DeviceType device_type = Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(task->thrd_id());
    if (device_type != DeviceType::kGPU) { continue; }
    int64_t device_id = Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(task->thrd_id());
    int64_t device_unique_id = GenDeviceUniqueId(machine_id, device_id);
    MemoryChain* mem_chain =
        &((*device2chain2mem_chain)[device_unique_id][task->task_set_info().chain_id()]);
    mem_chain->sorted_tasks.push_back(task);
    for (auto& pair : *(task->mutable_produced_regst_desc())) {
      RegstDescProto* regst_desc = &pair.second;
      if (regst_desc->mem_case().has_device_cuda_mem()
          && regst_desc->mem_case().device_cuda_mem().device_id() == device_id
          && regst_desc->enable_reuse_mem() && regst_desc->register_num() == 1
          && regst_desc->mem_block_id() == -1 && regst_desc->mem_block_offset() == -1
          && regst_desc->regst_desc_type().has_data_regst_desc()
          && Shape(regst_desc->regst_desc_type().data_regst_desc().time_shape()) == meta_shape) {
        CHECK(mem_chain->mem_reused_regsts.insert(regst_desc).second);
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
  for (MemoryChain* merged_chain : *merged_chains) {
    if (IsStrictOrderL2R(merged_chain, mem_chain)) {
      merged_chain->sorted_tasks.insert(merged_chain->sorted_tasks.end(),
                                        mem_chain->sorted_tasks.begin(),
                                        mem_chain->sorted_tasks.end());
      merged_chain->mem_reused_regsts.insert(mem_chain->mem_reused_regsts.begin(),
                                             mem_chain->mem_reused_regsts.end());
      return true;
    }
  }
  return false;
}

void GenMemChainTasksAndRegsts(
    Plan* plan, const PlanTaskGraph& plan_task_graph,
    HashMap<int64_t, std::vector<TaskProto*>>* mem_chain2sorted_tasks,
    HashMap<int64_t, HashSet<RegstDescProto*>>* mem_chain2mem_reused_regsts) {
  mem_chain2sorted_tasks->clear();
  mem_chain2mem_reused_regsts->clear();
  HashMap<int64_t, HashMap<int64_t, MemoryChain>> device2chain2mem_chain;
  InitMemoryChains(plan, &device2chain2mem_chain);

  auto IsStrictOrderL2R = [&](const MemoryChain* lhs, const MemoryChain* rhs) -> bool {
    const TaskProto* l_chain_sink_task_node = lhs->sorted_tasks.back();
    const TaskProto* r_chain_source_task_node = rhs->sorted_tasks.front();
    CHECK_LT(l_chain_sink_task_node->task_set_info().order_in_graph(),
             r_chain_source_task_node->task_set_info().order_in_graph());
    return plan_task_graph.IsReachable(l_chain_sink_task_node->task_id(),
                                       r_chain_source_task_node->task_id());
  };

  int64_t mem_chain_id = 0;

  for (auto& device_chain_pair : device2chain2mem_chain) {
    if (device_chain_pair.second.empty()) { continue; }
    // sort
    std::vector<MemoryChain*> mem_chains;
    std::vector<MemoryChain*> merged_chains;
    for (auto& pair : device_chain_pair.second) { mem_chains.push_back(&pair.second); }
    std::sort(mem_chains.begin(), mem_chains.end(), [&](MemoryChain* lhs, MemoryChain* rhs) {
      int64_t lhs_order_in_graph = lhs->sorted_tasks.front()->task_set_info().order_in_graph();
      int64_t rhs_order_in_graph = rhs->sorted_tasks.front()->task_set_info().order_in_graph();
      CHECK_NE(lhs_order_in_graph, rhs_order_in_graph);
      return lhs_order_in_graph < rhs_order_in_graph;
    });
    // merge
    for (MemoryChain* mem_chain : mem_chains) {
      if (!TryMergeMemChain2MergedChains(&merged_chains, mem_chain, IsStrictOrderL2R)) {
        merged_chains.push_back(mem_chain);
      }
    }
    for (MemoryChain* merged_chain : merged_chains) {
      std::vector<TaskProto*>* sorted_tasks = &((*mem_chain2sorted_tasks)[mem_chain_id]);
      CHECK(sorted_tasks->empty());
      sorted_tasks->insert(sorted_tasks->end(), merged_chain->sorted_tasks.begin(),
                           merged_chain->sorted_tasks.end());
      HashSet<RegstDescProto*>* mem_reused_regsts = &((*mem_chain2mem_reused_regsts)[mem_chain_id]);
      CHECK(mem_reused_regsts->empty());
      mem_reused_regsts->insert(merged_chain->mem_reused_regsts.begin(),
                                merged_chain->mem_reused_regsts.end());
      ++mem_chain_id;
    }
  }

  CHECK_EQ(mem_chain2sorted_tasks->size(), mem_chain2mem_reused_regsts->size());
}

void GenRegstApplyReleaseQueueAndRegstMutualExclusions(
    const std::vector<TaskProto*>* sorted_tasks, const HashSet<RegstDescProto*>* mem_reused_regsts,
    const HashMap<int64_t, RegstDescProto*>* regst_desc_id2regst_desc,
    std::vector<HashSet<RegstDescProto*>>* alloc_regsts_queue,
    std::vector<HashSet<RegstDescProto*>>* free_regsts_queue,
    HashMap<RegstDescProto*, HashSet<RegstDescProto*>>* regst2mutual_exclusion_regsts,
    HashMap<RegstDescProto*, RegstDescProto*>* consumer2inplaced_regst) {
  alloc_regsts_queue->clear();
  free_regsts_queue->clear();
  regst2mutual_exclusion_regsts->clear();
  consumer2inplaced_regst->clear();
  alloc_regsts_queue->resize(sorted_tasks->size());
  free_regsts_queue->resize(sorted_tasks->size());
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

  auto TryFindFirstInplacedRegstDesc = [&](RegstDescProto* consumer_regst) -> RegstDescProto* {
    RegstDescProto* inplaced_regst = nullptr;
    while (consumer_regst->has_hint_inplace_consumed_regst_desc_id()
           && consumer_regst->hint_inplace_consumed_regst_desc_id() != -1) {
      RegstDescProto* hint_inplaced_regst =
          regst_desc_id2regst_desc->at(consumer_regst->hint_inplace_consumed_regst_desc_id());
      if (mem_reused_regsts->find(hint_inplaced_regst) != mem_reused_regsts->end()) {
        inplaced_regst = hint_inplaced_regst;
        consumer_regst = hint_inplaced_regst;
      } else {
        break;
      }
    }
    return inplaced_regst;
  };

  HashMap<int64_t, int64_t> regst_desc_id2release_index;
  for (RegstDescProto* regst_desc : *mem_reused_regsts) {
    RegstDescProto* inplaced_regst_desc = TryFindFirstInplacedRegstDesc(regst_desc);
    if (inplaced_regst_desc != nullptr) {
      CHECK(consumer2inplaced_regst->emplace(regst_desc, inplaced_regst_desc).second);
      continue;
    }

    CHECK(alloc_regsts_queue->at(task_id2sorted_id.at(regst_desc->producer_task_id()))
              .insert(regst_desc)
              .second);
    CHECK(regst_desc_id2release_index
              .emplace(regst_desc->regst_desc_id(), FindLastReleaseIndexInSortedTasks(regst_desc))
              .second);
  }
  // inplace extend regst release index
  for (auto pair : *consumer2inplaced_regst) {
    RegstDescProto* consumer_regst_desc = pair.first;
    int64_t inplaced_regst_desc_id = pair.second->regst_desc_id();
    CHECK(regst_desc_id2release_index.find(inplaced_regst_desc_id)
          != regst_desc_id2release_index.end());
    regst_desc_id2release_index.at(inplaced_regst_desc_id) =
        std::max(regst_desc_id2release_index.at(inplaced_regst_desc_id),
                 FindLastReleaseIndexInSortedTasks(consumer_regst_desc));
  }
  for (const auto& pair : regst_desc_id2release_index) {
    CHECK(
        free_regsts_queue->at(pair.second).insert(regst_desc_id2regst_desc->at(pair.first)).second);
  }

  HashSet<RegstDescProto*> remain_regsts;
  for (int64_t i = 0; i < sorted_tasks->size(); ++i) {
    for (RegstDescProto* apply_regst : alloc_regsts_queue->at(i)) {
      CHECK(regst2mutual_exclusion_regsts->emplace(apply_regst, HashSet<RegstDescProto*>()).second);
      for (RegstDescProto* remain_regst : remain_regsts) {
        CHECK(regst2mutual_exclusion_regsts->at(apply_regst).insert(remain_regst).second);
        CHECK(regst2mutual_exclusion_regsts->at(remain_regst).insert(apply_regst).second);
      }
      CHECK(remain_regsts.insert(apply_regst).second);
    }
    for (RegstDescProto* release_regst : free_regsts_queue->at(i)) {
      CHECK_EQ(remain_regsts.erase(release_regst), 1);
    }
  }
  CHECK(remain_regsts.empty());
}

struct MemBlockResultInfo {
  int64_t mem_block_size;
  HashMap<RegstDescProto*, int64_t> regst_desc2offset;
};

struct Piece {
  int64_t begin;
  int64_t end;
  bool is_free;
};
using PieceIt = std::list<Piece>::iterator;

class MemBlockBuffer final {
 public:
  MemBlockBuffer(int64_t size) : buffer_size_(size) {
    Piece start_piece;
    start_piece.begin = 0;
    start_piece.end = size;
    start_piece.is_free = true;
    piece_list_.push_back(start_piece);
  };
  ~MemBlockBuffer() = default;

  void Occupy(int64_t begin, int64_t end);
  int64_t FindFreeOffset(int64_t size, int64_t* offset);

 private:
  void CheckValid() {
    CHECK(piece_list_.size() >= 1);
    CHECK(piece_list_.begin()->begin == 0);
    CHECK(std::prev(piece_list_.end())->end == buffer_size_);
    for (auto it = std::next(piece_list_.begin()); it != piece_list_.end(); ++it) {
      auto pre_it = std::prev(it);
      CHECK(pre_it->begin < pre_it->end && pre_it->end == it->begin);
    }
  }

  void MergePieceAndCheckValid() {
    CheckValid();
    for (auto it = std::next(piece_list_.begin()); it != piece_list_.end(); ++it) {
      auto pre_it = std::prev(it);
      if (it->is_free == pre_it->is_free) {
        it->begin = pre_it->begin;
        CHECK(piece_list_.erase(pre_it) == it);
      }
    }
    CheckValid();
  }

  std::list<Piece> piece_list_;
  int64_t buffer_size_;
};

void MemBlockBuffer::Occupy(int64_t begin, int64_t end) {
  CHECK(begin < end && end <= buffer_size_);
  for (auto it = piece_list_.begin(); it != piece_list_.end(); ++it) {
    if (it->end <= begin) { continue; }
    if (end <= it->begin) { break; }
    if (it->is_free) {
      if (begin != it->begin) {
        CHECK(it->begin < begin);
        CHECK(begin < it->end);
        Piece free_piece;
        free_piece.begin = it->begin;
        free_piece.end = begin;
        free_piece.is_free = true;
        it->begin = begin;
        it = piece_list_.insert(it, free_piece);
      } else if (end < it->end) {
        Piece busy_piece;
        busy_piece.begin = it->begin;
        busy_piece.end = end;
        busy_piece.is_free = false;
        it->begin = end;
        it = piece_list_.insert(it, busy_piece);
        begin = end;
      } else {
        it->is_free = false;
        begin = it->end;
      }
    } else {
      begin = it->end;
      end = std::max(begin, end);
    }
  }
  MergePieceAndCheckValid();
}

int64_t MemBlockBuffer::FindFreeOffset(int64_t size, int64_t* offset) {
  CheckValid();
  for (auto it = piece_list_.begin(); it != piece_list_.end(); ++it) {
    if (it->is_free && (it->end - it->begin) >= size) {
      *offset = it->begin;
      return buffer_size_;
    }
  }
  auto last_it = std::prev(piece_list_.end());
  if (last_it->is_free) {
    *offset = last_it->begin;
    return buffer_size_ + size - (last_it->end - last_it->begin);
  } else {
    *offset = buffer_size_;
    return buffer_size_ + size;
  }
}

void MemReusedAlgorithm0_OfColorImproved(
    const HashMap<RegstDescProto*, HashSet<RegstDescProto*>>& regst2mutual_exclusion_regsts,
    MemBlockResultInfo* result) {
  HashMap<RegstDescProto*, int64_t>* regst_desc2offset = &(result->regst_desc2offset);
  regst_desc2offset->clear();
  std::vector<RegstDescProto*> order;
  int64_t buffer_size = 256;
  HashMap<RegstDescProto*, int64_t> regst_desc2size;
  for (const auto& pair : regst2mutual_exclusion_regsts) {
    order.push_back(pair.first);
    CHECK(regst_desc2size.emplace(pair.first, RtRegstDesc(*pair.first).TotalMainByteSize4AllRegst())
              .second);
  }
  std::sort(order.begin(), order.end(), [&](RegstDescProto* lhs, RegstDescProto* rhs) {
    return regst_desc2size.at(lhs) > regst_desc2size.at(rhs);
  });
  for (RegstDescProto* regst_desc : order) {
    MemBlockBuffer buffer(buffer_size);
    for (RegstDescProto* mutual_regst : regst2mutual_exclusion_regsts.at(regst_desc)) {
      if (regst_desc2offset->find(mutual_regst) != regst_desc2offset->end()) {
        int64_t begin = regst_desc2offset->at(mutual_regst);
        int64_t end = begin + regst_desc2size.at(mutual_regst);
        buffer.Occupy(begin, end);
      }
    }
    int64_t offset = -1;
    buffer_size = buffer.FindFreeOffset(regst_desc2size.at(regst_desc), &offset);
    CHECK(offset >= 0 && offset < buffer_size);
    CHECK(regst_desc2offset->emplace(regst_desc, offset).second);
  }
  result->mem_block_size = buffer_size;
}

class BfcAllocator final {
 public:
  BfcAllocator(int64_t size) : buffer_size_(size) {
    Piece start_piece;
    start_piece.begin = 0;
    start_piece.end = size;
    start_piece.is_free = true;
    piece_list_.push_back(start_piece);
  };
  ~BfcAllocator() = default;

  int64_t AllocateRaw(int64_t size);
  void DeallocateRaw(int64_t offset, int64_t size);
  int64_t buffer_size() const { return buffer_size_; }

 private:
  void CheckValid() {
    CHECK(piece_list_.size() >= 1);
    CHECK(piece_list_.front().begin == 0);
    CHECK(piece_list_.back().end == buffer_size_);
    for (auto it = std::next(piece_list_.begin()); it != piece_list_.end(); ++it) {
      auto pre_it = std::prev(it);
      CHECK(pre_it->begin < pre_it->end && pre_it->end == it->begin);
      CHECK(!(pre_it->is_free && it->is_free));
    }
  }

  void MergeFreePieceAndCheckValid() {
    for (auto it = std::next(piece_list_.begin()); it != piece_list_.end(); ++it) {
      auto pre_it = std::prev(it);
      if (it->is_free && pre_it->is_free) {
        it->begin = pre_it->begin;
        CHECK(piece_list_.erase(pre_it) == it);
      }
    }
    CheckValid();
  }

  std::list<Piece> piece_list_;
  int64_t buffer_size_;
  HashMap<int64_t, PieceIt> offset2occupied_piece_;
};

int64_t BfcAllocator::AllocateRaw(int64_t size) {
  int64_t offset = -1;
  PieceIt candidate_piece = piece_list_.end();
  for (auto it = piece_list_.begin(); it != piece_list_.end(); ++it) {
    int64_t piece_size = it->end - it->begin;
    if (it->is_free && piece_size >= size
        && (candidate_piece == piece_list_.end()
            || piece_size < (candidate_piece->end - candidate_piece->begin))) {
      candidate_piece = it;
    }
  }
  if (candidate_piece == piece_list_.end()) {
    auto last_it = std::prev(piece_list_.end());
    if (last_it->is_free) {
      offset = last_it->begin;
      buffer_size_ += size - (last_it->end - last_it->begin);
      last_it->end = buffer_size_;
      last_it->is_free = false;
      CHECK(offset2occupied_piece_.emplace(offset, last_it).second);
    } else {
      offset = last_it->end;
      buffer_size_ += size;
      Piece new_piece;
      new_piece.begin = last_it->end;
      new_piece.end = buffer_size_;
      new_piece.is_free = false;
      piece_list_.push_back(new_piece);
      CHECK(offset2occupied_piece_.emplace(offset, std::prev(piece_list_.end())).second);
    }
  } else {
    int64_t piece_size = candidate_piece->end - candidate_piece->begin;
    offset = candidate_piece->begin;
    if (piece_size > size) {
      Piece new_piece;
      new_piece.begin = candidate_piece->begin;
      new_piece.end = candidate_piece->begin + size;
      new_piece.is_free = false;
      candidate_piece->begin = new_piece.end;
      PieceIt new_it = piece_list_.insert(candidate_piece, new_piece);
      CHECK(offset2occupied_piece_.emplace(offset, new_it).second);
    } else {
      CHECK_EQ(size, piece_size);
      candidate_piece->is_free = false;
      CHECK(offset2occupied_piece_.emplace(offset, candidate_piece).second);
    }
  }
  CheckValid();
  CHECK_NE(offset, -1);
  CHECK(offset2occupied_piece_.find(offset) != offset2occupied_piece_.end());
  return offset;
}

void BfcAllocator::DeallocateRaw(int64_t offset, int64_t size) {
  CHECK(offset2occupied_piece_.find(offset) != offset2occupied_piece_.end());
  PieceIt occupied_piece = offset2occupied_piece_.at(offset);
  CHECK(occupied_piece->is_free == false);
  CHECK_EQ((occupied_piece->end - occupied_piece->begin), size);
  occupied_piece->is_free = true;
  CHECK(offset2occupied_piece_.erase(offset) == 1);
  MergeFreePieceAndCheckValid();
}

void MemReusedAlgorithm1_TfBfcImproved(
    const std::vector<HashSet<RegstDescProto*>>& alloc_regsts_queue,
    const std::vector<HashSet<RegstDescProto*>>& free_regsts_queue, MemBlockResultInfo* result) {
  HashMap<RegstDescProto*, int64_t>* regst_desc2offset = &(result->regst_desc2offset);
  regst_desc2offset->clear();
  int64_t buffer_size = 256;
  BfcAllocator bfc_allocator(buffer_size);

  auto GetRegstSize = [](const RegstDescProto* regst) -> int64_t {
    return RtRegstDesc(*regst).TotalMainByteSize4AllRegst();
  };

  CHECK_EQ(alloc_regsts_queue.size(), free_regsts_queue.size());
  for (int64_t i = 0; i < alloc_regsts_queue.size(); ++i) {
    for (RegstDescProto* apply_regst : alloc_regsts_queue.at(i)) {
      CHECK(regst_desc2offset
                ->emplace(apply_regst, bfc_allocator.AllocateRaw(GetRegstSize(apply_regst)))
                .second);
    }
    for (RegstDescProto* release_regst : free_regsts_queue.at(i)) {
      CHECK(regst_desc2offset->find(release_regst) != regst_desc2offset->end());
      bfc_allocator.DeallocateRaw(regst_desc2offset->at(release_regst),
                                  GetRegstSize(release_regst));
    }
  }
  result->mem_block_size = bfc_allocator.buffer_size();
}

void MemReusedAlgorithm2_MinOrderGrowth(
    const HashMap<RegstDescProto*, HashSet<RegstDescProto*>>& regst2mutual_exclusion_regsts,
    MemBlockResultInfo* result) {
  // idea by LiKe
  TODO();
}

void SelectAlgorithmGenMemBlockOffset4Regsts(
    int64_t algo_id, const std::vector<HashSet<RegstDescProto*>>& alloc_regsts_queue,
    const std::vector<HashSet<RegstDescProto*>>& free_regsts_queue,
    const HashMap<RegstDescProto*, HashSet<RegstDescProto*>>& regst2mutual_exclusion_regsts,
    MemBlockResultInfo* result) {
  switch (algo_id) {
    case 0: MemReusedAlgorithm0_OfColorImproved(regst2mutual_exclusion_regsts, result); break;
    case 1: MemReusedAlgorithm1_TfBfcImproved(alloc_regsts_queue, free_regsts_queue, result); break;
    case 2: MemReusedAlgorithm2_MinOrderGrowth(regst2mutual_exclusion_regsts, result); break;
    default: UNIMPLEMENTED();
  }
}

}  // namespace

void IntraJobMemSharingUtil::InferMemBlockId4MemReusedRegst(Plan* plan,
                                                            const PlanTaskGraph& plan_task_graph) {
  // 1 device 1 mem chain
  HashMap<int64_t, std::vector<TaskProto*>> mem_chain2sorted_tasks;
  HashMap<int64_t, HashSet<RegstDescProto*>> mem_chain2mem_reused_regsts;
  GenMemChainTasksAndRegsts(plan, plan_task_graph, &mem_chain2sorted_tasks,
                            &mem_chain2mem_reused_regsts);
  if (mem_chain2mem_reused_regsts.empty()) { return; }
  HashMap<int64_t, RegstDescProto*> regst_desc_id2regst_desc;
  GenRegstDescId2RegstDesc(plan, &regst_desc_id2regst_desc);
  // info for algorithm
  HashMap<int64_t, std::vector<HashSet<RegstDescProto*>>> mem_chain2task2apply_regsts;
  HashMap<int64_t, std::vector<HashSet<RegstDescProto*>>> mem_chain2task2release_regsts;
  HashMap<int64_t, HashMap<RegstDescProto*, HashSet<RegstDescProto*>>>
      mem_chain2regst2mutual_exclusion_regsts;
  // info for inplace
  HashMap<int64_t, HashMap<RegstDescProto*, RegstDescProto*>> mem_chain2consumer2inplaced_regst;

  // step 1: generate regst apply/release queue AND regst mutual exclusions
  for (const auto& pair : mem_chain2mem_reused_regsts) {
    GenRegstApplyReleaseQueueAndRegstMutualExclusions(
        &mem_chain2sorted_tasks.at(pair.first), &pair.second, &regst_desc_id2regst_desc,
        &mem_chain2task2apply_regsts[pair.first], &mem_chain2task2release_regsts[pair.first],
        &mem_chain2regst2mutual_exclusion_regsts[pair.first],
        &mem_chain2consumer2inplaced_regst[pair.first]);
  }

  // step 2: multi-thread run several algorithm for each mem chain
  HashMap<int64_t, std::vector<MemBlockResultInfo>> mem_chain2results;
  {
    int64_t cpu_num = std::thread::hardware_concurrency();
    int64_t work_size = mem_chain2mem_reused_regsts.size() * kMemReuseAlgorithmNum;
    int64_t thread_pool_size = std::min<int64_t>(work_size, cpu_num);
    BlockingCounter counter(work_size);
    ThreadPool thread_pool(thread_pool_size);
    for (const auto& pair : mem_chain2mem_reused_regsts) {
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
    CHECK(!mem_chain2mem_reused_regsts.at(pair.first).empty());
    CHECK(!mem_chain2sorted_tasks.at(pair.first).empty());
    const std::vector<MemBlockResultInfo>& results = pair.second;
    int64_t best_algo_id = 0;
    for (int64_t algo_id = 0; algo_id < kMemReuseAlgorithmNum; ++algo_id) {
      CHECK_GT(results.at(algo_id).mem_block_size, 0);
      CHECK(!results.at(algo_id).regst_desc2offset.empty());
      if (results.at(algo_id).mem_block_size < results.at(best_algo_id).mem_block_size) {
        best_algo_id = algo_id;
      }
    }
    LOG(INFO) << "mem reuse algorithm choose result : algorithm " << best_algo_id
              << " is best in mem_chain " << pair.first
              << " use mem_block_size =  " << results.at(best_algo_id).mem_block_size;
    int64_t mem_block_id = Global<IDMgr>::Get()->NewMemBlockId();
    const MemBlockResultInfo& best_result = results.at(best_algo_id);
    CHECK_EQ(mem_chain2mem_reused_regsts.at(pair.first).size(),
             (best_result.regst_desc2offset.size()
              + mem_chain2consumer2inplaced_regst.at(pair.first).size()));
    for (const auto& regst_offset_pair : best_result.regst_desc2offset) {
      RegstDescProto* regst_desc = regst_offset_pair.first;
      int64_t offset = regst_offset_pair.second;
      CHECK_EQ(regst_desc->mem_block_id(), -1);
      regst_desc->set_mem_block_id(mem_block_id);
      regst_desc->set_mem_block_offset(offset);
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
  }
}

}  // namespace oneflow
