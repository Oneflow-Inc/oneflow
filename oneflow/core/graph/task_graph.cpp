#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/normal_backward_compute_task_node.h"
#include "oneflow/core/graph/normal_model_update_compute_task_node.h"
#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/reduce_global_add_compute_task_node.h"
#include "oneflow/core/graph/reduce_gather_compute_task_node.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/thrd_id_generator.h"

namespace oneflow {

TaskGraph::TaskGraph(std::unique_ptr<const LogicalGraph>&& logical_gph) {
  logical_gph_ = std::move(logical_gph);
  HashMap<const LogicalNode*, std::vector<CompTaskNode*>> logical2sorted_comp_tasks;
  HashMap<const LogicalNode*, std::vector<TaskNode*>> logical2sorted_in_box;
  HashMap<const LogicalNode*, std::vector<TaskNode*>> logical2sorted_out_box;
  HashMap<CompTaskNode*, HashMap<int64_t, std::vector<TaskNode*>>> buf_task;
  const JobDesc* job_desc = Global<JobDesc>::Get();
  auto MutBufTask = [&](CompTaskNode* task_node, int64_t machine_id, int32_t mem_zone_id) {
    auto& buf_vec = buf_task[task_node][machine_id];
    if (buf_vec.empty()) { buf_vec.assign(job_desc->MemZoneNum(), nullptr); }
    return &(buf_vec.at(mem_zone_id));
  };

  std::vector<int64_t> cpu_device_offset(job_desc->TotalMachineNum(), 0);
  auto AllocateCpuThrdIdEvenly = [&](const TaskNode* task_node) {
    CHECK(!task_node->IsPersistence());
    int64_t ret = -1;
    int64_t& offset = cpu_device_offset.at(task_node->machine_id());
    ret = Global<IDMgr>::Get()->GetCpuDeviceThrdId(offset);
    offset = (offset + 1) % job_desc->CpuDeviceNum();
    return ret;
  };

  std::vector<std::pair<int64_t, CompTaskNode*>> machine_persistence_task_vec;
  logical_gph_->ForEachNode([&](const LogicalNode* logical_node) {
    logical_node->GenSortedCompTaskNodes(
        AllocateCpuThrdIdEvenly, &machine_persistence_task_vec, [&](CompTaskNode* comp_task_node) {
          AddAllocatedNode(comp_task_node);
          logical2sorted_comp_tasks[logical_node].push_back(comp_task_node);
          comp_task_node->set_area_id(logical_node->GetAreaId());
        });
  });
  GeneratePersistenceThrdId(machine_persistence_task_vec);
  logical_gph_->ForEachEdge([&](const LogicalEdge* logical_edge) {
    BldSubTskGphMthd method =
        GetMthdForBldSubTskGph(logical_edge->src_node(), logical_edge->dst_node());
    (this->*method)(logical_edge->src_node(), logical_edge->dst_node(),
                    logical2sorted_comp_tasks.at(logical_edge->src_node()),
                    logical2sorted_comp_tasks.at(logical_edge->dst_node()), &logical2sorted_in_box,
                    &logical2sorted_out_box, MutBufTask, AllocateCpuThrdIdEvenly);
    SetAreaIdForNewNodes(logical_edge->src_node(), logical_edge->dst_node());
  });
  MergeChainAndSetOrderInGraphForEachNode();
  ToDotWithAutoFilePath();
}

void TaskGraph::GeneratePersistenceThrdId(
    const std::vector<std::pair<int64_t, CompTaskNode*>>& persistence_nodes) {
  std::vector<std::pair<int64_t, TaskType>> machine_task_type_vec;
  for (auto pair : persistence_nodes) {
    machine_task_type_vec.emplace_back(std::make_pair(pair.first, pair.second->GetTaskType()));
  }

  ThrdIdGenerator generator(machine_task_type_vec, Global<IDMgr>::Get()->BasePersistenceThrdId());
  for (const auto pair : persistence_nodes) {
    int64_t thrd_id = generator.GenerateThrdId(pair.first, pair.second->GetTaskType());
    pair.second->set_thrd_id(thrd_id);
  }
}

void TaskGraph::AcyclicTopoForEachNode(std::function<bool(TaskNode* node)> IsAllowedStartNode,
                                       std::function<void(TaskNode* node)> Handler) const {
  std::list<TaskNode*> starts;
  ForEachNode([&](TaskNode* node) {
    if (node->in_edges().empty() && IsAllowedStartNode(node)) { starts.push_back(node); }
  });
  auto ForEachInNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](TaskNode* node_on_in_edge) {
      if (IsBackEdge(node_on_in_edge, node)) return;
      Handler(const_cast<TaskNode*>(node_on_in_edge));
    });
  };
  auto ForEachOutNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](TaskNode* node_on_out_edge) {
      if (IsBackEdge(node, node_on_out_edge)) return;
      Handler(const_cast<TaskNode*>(node_on_out_edge));
    });
  };
  // DfsTopo will cause inappropriate chain graph
  TopoForEachNode(starts, ForEachInNode, ForEachOutNode, Handler);
}

void TaskGraph::AcyclicTopoForEachNode(std::function<void(TaskNode* node)> Handler) const {
  return AcyclicTopoForEachNode([](TaskNode*) { return true; }, Handler);
}

void TaskGraph::RemoveEmptyRegsts() {
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeProducedBlob(); });
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeConsumedRegst(); });
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeProducedRegst(); });
  ForEachNode([&](TaskNode* node) { node->UnbindBnWithEmptyRegst(); });
}

void TaskGraph::AddOrderingCtrlEdgeInSameChain() { BuildCtrlRegstDescInSameChain(); }

void TaskGraph::MergeChainAndSetOrderInGraphForEachNode() {
  ChainGraph chain_graph(*this);
  const auto& ordered_chain_nodes = chain_graph.OrderdedChainNodes();
  int64_t order_in_graph = 0;
  for (auto& chain_node : ordered_chain_nodes) {
    auto& ordered_in_chain = chain_node->TaskNodes();
    int64_t chain_id = chain_node->chain_id();
    for (auto& task_node : ordered_in_chain) {
      task_node->set_chain_id(chain_id);
      task_node->set_order_in_graph(order_in_graph);
      ordered_task_nodes_.emplace_back(task_node);
      ++order_in_graph;
    }
  }
}

void TaskGraph::BuildCtrlRegstDescInSameChain() {
  HashMap<int64_t, TaskNode*> chain_id2node;
  for (auto node : ordered_task_nodes_) {
    int64_t chain_id = node->chain_id();
    auto iter = chain_id2node.find(chain_id);
    if (iter == chain_id2node.end()) {
      CHECK(chain_id2node.emplace(chain_id, node).second);
    } else {
      iter->second->BuildCtrlRegstDescIfNeed(node);
      iter->second = node;
    }
  }
}

void TaskGraph::EnableMemSharingInReduceStruct() {
  std::unordered_set<ReduceTaskNodes, ReduceTaskNodesHasher> reduce_tasks;
  CollectReduceTaskNodes(&reduce_tasks);
  for (auto& reduce_task : reduce_tasks) {
    EnableMemSharingInOneReduce(reduce_task);
    AddCtrlEdge4MemSharingInOneReduce(reduce_task);
  }
}

void TaskGraph::CollectReduceTaskNodes(
    std::unordered_set<ReduceTaskNodes, ReduceTaskNodesHasher>* reduce_tasks) const {
  auto FindSuccReduceTaskNode = [](CompTaskNode* task_node, TaskType type) -> CompTaskNode* {
    for (TaskEdge* out_edge : task_node->out_edges()) {
      TaskNode* dst_node = out_edge->dst_node();
      if (dst_node->GetTaskType() == type) { return dynamic_cast<CompTaskNode*>(dst_node); }
    }
    return nullptr;
  };

  auto FindConcatAndScatter = [&](CompTaskNode* bw_or_md_diff_acc,
                                  ReduceTaskNodes* reduce_task_nodes) {
    CompTaskNode* concat_task_node =
        FindSuccReduceTaskNode(bw_or_md_diff_acc, TaskType::kReduceConcat);
    if (concat_task_node != nullptr) {
      reduce_task_nodes->concat = concat_task_node;
      reduce_task_nodes->scatter =
          FindSuccReduceTaskNode(reduce_task_nodes->concat, TaskType::kReduceScatter);
    } else {
      reduce_task_nodes->scatter =
          FindSuccReduceTaskNode(bw_or_md_diff_acc, TaskType::kReduceScatter);
    }
  };

  ForEachNode([&](TaskNode* task_node) {
    if (IsBackwardTaskType(task_node->GetTaskType()) == false) { return; }
    if (task_node->device_type() != DeviceType::kGPU) { return; }
    CompTaskNode* bw_task_node = dynamic_cast<CompTaskNode*>(task_node);
    CHECK(bw_task_node != nullptr);
    if (bw_task_node->logical_node()->HasOpWithModelBlob() == false) { return; }
    if (bw_task_node->parallel_ctx()->policy() != kDataParallel
        || bw_task_node->parallel_ctx()->parallel_num() < 2) {
      return;
    }

    ReduceTaskNodes reduce_task_nodes;
    CompTaskNode* diff_acc_task_node = FindSuccReduceTaskNode(bw_task_node, TaskType::kMdDiffAcc);
    if (diff_acc_task_node != nullptr) {
      FindConcatAndScatter(diff_acc_task_node, &reduce_task_nodes);
    } else {
      FindConcatAndScatter(bw_task_node, &reduce_task_nodes);
    }
    CompTaskNode* local_add_task_node =
        FindSuccReduceTaskNode(reduce_task_nodes.scatter, TaskType::kReduceLocalAdd);
    if (local_add_task_node != nullptr) {
      reduce_task_nodes.local_add = local_add_task_node;
      reduce_task_nodes.global_add =
          FindSuccReduceTaskNode(reduce_task_nodes.local_add, TaskType::kReduceGlobalAdd);
    } else {
      reduce_task_nodes.global_add =
          FindSuccReduceTaskNode(reduce_task_nodes.scatter, TaskType::kReduceGlobalAdd);
    }
    reduce_task_nodes.gather =
        FindSuccReduceTaskNode(reduce_task_nodes.global_add, TaskType::kReduceGather);
    reduce_task_nodes.split =
        FindSuccReduceTaskNode(reduce_task_nodes.gather, TaskType::kReduceSplit);
    if (reduce_task_nodes.split == nullptr) { CHECK(reduce_task_nodes.concat == nullptr); }

    CHECK(reduce_task_nodes.scatter != nullptr);
    CHECK(reduce_task_nodes.global_add != nullptr);
    CHECK(reduce_task_nodes.gather != nullptr);
    reduce_tasks->insert(reduce_task_nodes);
  });
}

void TaskGraph::EnableMemSharingInReduceConcatSplitIfNeed(
    const ReduceTaskNodes& reduce_task_nodes,
    std::function<void(RegstDesc*, int64_t)> SetMemSharedField4Regst) {
  if (reduce_task_nodes.concat == nullptr) { return; }
  int32_t reduce_num = reduce_task_nodes.split->produced_regsts().size();

  std::shared_ptr<RegstDesc> concat_out_regst = reduce_task_nodes.concat->GetProducedRegst("out");
  std::shared_ptr<RegstDesc> split_in_regst = reduce_task_nodes.split->GetSoleConsumedRegst("in");
  const BlobDesc* concat_out_packed = concat_out_regst->GetBlobDesc(GenPackedLbi());
  const BlobDesc* split_in_packed = split_in_regst->GetBlobDesc(GenPackedLbi());
  size_t concat_out_byte_size = RtBlobDesc(*concat_out_packed).ByteSizeOfBlobBody();
  size_t split_in_byte_size = RtBlobDesc(*split_in_packed).ByteSizeOfBlobBody();
  CHECK_EQ(concat_out_byte_size, split_in_byte_size);
  SetMemSharedField4Regst(concat_out_regst.get(), 0);
  SetMemSharedField4Regst(split_in_regst.get(), 0);

  int64_t offset = 0;
  FOR_RANGE(int32_t, idx, 0, reduce_num) {
    auto concat_in_regst =
        reduce_task_nodes.concat->GetSoleConsumedRegst("in_" + std::to_string(idx));
    auto split_out_regst = reduce_task_nodes.split->GetProducedRegst("out_" + std::to_string(idx));
    SetMemSharedField4Regst(concat_in_regst.get(), offset);
    SetMemSharedField4Regst(split_out_regst.get(), offset);

    // Check shape invariant
    const BlobDesc* concat_in_packed = concat_in_regst->GetBlobDesc(GenPackedLbi());
    const BlobDesc* split_out_packed = split_out_regst->GetBlobDesc(GenPackedLbi());
    size_t concat_in_byte_size = RtBlobDesc(*concat_in_packed).ByteSizeOfBlobBody();
    size_t split_out_byte_size = RtBlobDesc(*split_out_packed).ByteSizeOfBlobBody();
    CHECK_EQ(concat_in_byte_size, split_out_byte_size);

    offset += concat_in_byte_size;
  }
}

void TaskGraph::EnableMemSharingInOneReduce(const ReduceTaskNodes& reduce_task_nodes) {
  std::shared_ptr<const ParallelDesc> parallel_desc =
      reduce_task_nodes.scatter->logical_node()->parallel_desc();
  int64_t parallel_num = parallel_desc->parallel_num();
  int64_t dev_num_of_each_machine = parallel_desc->device_num_of_each_machine();
  int64_t machine_num = parallel_desc->sorted_machine_ids().size();
  CHECK_EQ(parallel_num, machine_num * dev_num_of_each_machine);
  int64_t parallel_id = reduce_task_nodes.scatter->parallel_ctx()->parallel_id();

  int64_t mem_shared_id = Global<IDMgr>::Get()->NewMemSharedId();
  std::vector<int64_t> blob_index2offset(parallel_num, 0);

  auto SetMemSharedField4Regst = [&](RegstDesc* regst, int64_t offset) {
    regst->set_enable_mem_sharing(true);
    regst->set_mem_shared_id(mem_shared_id);
    regst->set_mem_shared_offset(offset);
  };

  EnableMemSharingInReduceConcatSplitIfNeed(reduce_task_nodes, SetMemSharedField4Regst);

  // scatter
  {
    std::shared_ptr<RegstDesc> consumed_regst =
        reduce_task_nodes.scatter->GetSoleConsumedRegst("in");
    consumed_regst->set_enable_mem_sharing(true);
    consumed_regst->set_mem_shared_id(mem_shared_id);
    consumed_regst->set_mem_shared_offset(0);
    int64_t total_model_byte_size =
        RtBlobDesc(*(consumed_regst->GetBlobDesc(GenPackedLbi()))).ByteSizeOfDataContentField();
    CHECK_EQ(0, total_model_byte_size % parallel_num);
    for (int64_t i = 0; i < parallel_num; ++i) {
      blob_index2offset.at(i) = total_model_byte_size / parallel_num * i;
    }

    for (int64_t i = 0; i < parallel_num; ++i) {
      SetMemSharedField4Regst(
          reduce_task_nodes.scatter->GetProducedRegst("out_" + std::to_string(i)).get(),
          blob_index2offset.at(i));
    }
  }

  auto SetOrCheck4ConsumedRegst = [&](RegstDesc* consumed_regst, bool is_inplace_regst,
                                      int64_t blob_id) {
    if (is_inplace_regst) {
      CHECK_EQ(mem_shared_id, consumed_regst->mem_shared_id());
      CHECK_EQ(blob_index2offset.at(blob_id), consumed_regst->mem_shared_offset());
    } else {
      SetMemSharedField4Regst(consumed_regst, blob_index2offset.at(blob_id));
    }
  };

  // local_add
  if (reduce_task_nodes.local_add) {
    HashSet<int64_t> inplace_blob_ids;
    int64_t dev_index_of_this_machine = parallel_id % dev_num_of_each_machine;
    for (int64_t i = dev_index_of_this_machine; i < parallel_num; i += dev_num_of_each_machine) {
      inplace_blob_ids.emplace(i);
    }

    ExecNode* local_add_exec_node = reduce_task_nodes.local_add->exec_gph().SoleNode();
    CHECK_EQ(parallel_num, local_add_exec_node->op()->input_bns().size());
    for (int64_t i = 0; i < parallel_num; ++i) {
      RegstDesc* consumed_regst =
          local_add_exec_node->RegstDesc4BnInOp(local_add_exec_node->op()->input_bns().Get(i));
      SetOrCheck4ConsumedRegst(consumed_regst, inplace_blob_ids.find(i) != inplace_blob_ids.end(),
                               i);
    }

    for (int64_t i = 0; i < machine_num; ++i) {
      SetMemSharedField4Regst(
          reduce_task_nodes.local_add->GetProducedRegst("out_" + std::to_string(i)).get(),
          blob_index2offset.at(i * dev_num_of_each_machine + dev_index_of_this_machine));
    }
  }

  auto HandleMemSharedFieldOfConsumedRegsts = [&](CompTaskNode* task_node,
                                                  int64_t consumed_regst_num) {
    auto& consumed_regsts = task_node->consumed_regsts();
    CHECK_EQ(consumed_regst_num, consumed_regsts.size());
    for (const auto& kv : consumed_regsts) {
      int64_t in_parallel_id = oneflow_cast<int64_t>(kv.first.substr(3));
      CHECK_EQ(1, kv.second.size());
      SetOrCheck4ConsumedRegst(kv.second.front().get(), in_parallel_id == parallel_id,
                               in_parallel_id);
    }
  };
  // global add
  int consumed_regst_num = reduce_task_nodes.local_add ? machine_num : parallel_num;
  HandleMemSharedFieldOfConsumedRegsts(reduce_task_nodes.global_add, consumed_regst_num);
  SetMemSharedField4Regst(reduce_task_nodes.global_add->GetProducedRegst("out").get(),
                          blob_index2offset.at(parallel_id));

  // gather
  HandleMemSharedFieldOfConsumedRegsts(reduce_task_nodes.gather, parallel_num);
  SetMemSharedField4Regst(reduce_task_nodes.gather->GetProducedRegst("out").get(),
                          blob_index2offset.at(0));
}

void TaskGraph::AddCtrlEdge4MemSharingInOneReduce(const ReduceTaskNodes& reduce_task_nodes) {
  std::shared_ptr<const ParallelDesc> parallel_desc =
      reduce_task_nodes.scatter->logical_node()->parallel_desc();
  int64_t parallel_num = parallel_desc->parallel_num();
  int64_t machine_num = parallel_desc->sorted_machine_ids().size();

  if (reduce_task_nodes.local_add == nullptr) {
    BuildCtrlRegstBetweenReduceCopyNodes(reduce_task_nodes.scatter, reduce_task_nodes.global_add,
                                         parallel_num - 1);
  } else {
    BuildCtrlRegstBetweenReduceCopyNodes(reduce_task_nodes.scatter, reduce_task_nodes.local_add,
                                         parallel_num - machine_num);
    BuildCtrlRegstBetweenReduceCopyNodes(reduce_task_nodes.local_add, reduce_task_nodes.global_add,
                                         machine_num - 1);
  }

  // global_add -> gather
  CHECK_EQ(2, reduce_task_nodes.global_add->out_edges().size());
  TaskNode* global_add_copy_d2h = nullptr;
  for (TaskEdge* out_edge : reduce_task_nodes.global_add->out_edges()) {
    if (out_edge->dst_node()->GetTaskType() == TaskType::kCopyHd) {
      global_add_copy_d2h = out_edge->dst_node();
    }
  }

  for (TaskEdge* in_edge : reduce_task_nodes.gather->in_edges()) {
    if (in_edge->src_node()->GetTaskType() == TaskType::kCopyHd) {
      global_add_copy_d2h->BuildCtrlRegstDesc(in_edge->src_node());
    }
  }
}

void TaskGraph::BuildCtrlRegstBetweenReduceCopyNodes(const CompTaskNode* src_reduce,
                                                     const CompTaskNode* dst_reduce,
                                                     int64_t copy_node_num) {
  struct ReduceCopyNodePair {
    TaskNode* copy_h2d;
    TaskNode* copy_d2h;
    ReduceCopyNodePair() : copy_h2d(nullptr), copy_d2h(nullptr) {}
  };
  HashMap<int64_t, ReduceCopyNodePair> mem_shared_offset2copy_nodes;

  for (TaskEdge* out_edge : src_reduce->out_edges()) {
    if (out_edge->dst_node()->GetTaskType() == TaskType::kCopyHd) {
      int64_t offset = out_edge->GetSoleRegst()->mem_shared_offset();
      mem_shared_offset2copy_nodes[offset].copy_d2h = out_edge->dst_node();
    }
  }
  CHECK_EQ(copy_node_num, mem_shared_offset2copy_nodes.size());

  for (TaskEdge* in_edge : dst_reduce->in_edges()) {
    if (in_edge->src_node()->GetTaskType() == TaskType::kCopyHd) {
      int64_t offset = in_edge->GetSoleRegst()->mem_shared_offset();
      CHECK(mem_shared_offset2copy_nodes.find(offset) != mem_shared_offset2copy_nodes.end());
      mem_shared_offset2copy_nodes.at(offset).copy_h2d = in_edge->src_node();
    }
  }

  for (const auto& kv : mem_shared_offset2copy_nodes) {
    kv.second.copy_d2h->BuildCtrlRegstDesc(kv.second.copy_h2d);
  }
}

void TaskGraph::AddCtrlEdgeInReduceStruct() {
  int64_t total_machine_num = Global<JobDesc>::Get()->resource().machine().size();
  if (total_machine_num == 1) { return; }

  AddCtrlEdgeForReduceTaskNode<ReduceGlobalAddLogicalNode, ReduceGlobalAddCompTaskNode>(
      total_machine_num);
  AddCtrlEdgeForReduceTaskNode<ReduceGatherLogicalNode, ReduceGatherCompTaskNode>(
      total_machine_num);
}

template<typename LogicalNodeType, typename TaskNodeType>
void TaskGraph::AddCtrlEdgeForReduceTaskNode(int64_t total_machine_num) {
  HashMap<const LogicalNodeType*, HashMap<int64_t, std::vector<TaskNodeType*>>>
      machine_id2reduce_task_nodes4same_logical_node;
  ForEachNode([&](TaskNode* task_node) {
    TaskNodeType* reduce_task_node = dynamic_cast<TaskNodeType*>(task_node);
    if (reduce_task_node != nullptr) {
      const LogicalNodeType* logical_node =
          dynamic_cast<const LogicalNodeType*>(reduce_task_node->logical_node());
      CHECK(logical_node != nullptr);
      machine_id2reduce_task_nodes4same_logical_node[logical_node][reduce_task_node->machine_id()]
          .push_back(reduce_task_node);
    }
  });

  for (const auto& kv : machine_id2reduce_task_nodes4same_logical_node) {
    const auto& machine_id2reduce_task_nodes = kv.second;
    if (machine_id2reduce_task_nodes.size() == 1) { continue; }
    for (int64_t machine_id = 0; machine_id < machine_id2reduce_task_nodes.size(); ++machine_id) {
      std::vector<std::pair<CopyCommNetTaskNode*, int64_t>> commnet_nodes_with_sort_val;
      CollectCopyCommNetForReduceTaskNodes(machine_id2reduce_task_nodes.at(machine_id),
                                           &commnet_nodes_with_sort_val);

      std::vector<int64_t> machine_id2sort_order(total_machine_num);
      for (size_t i = 0; i < total_machine_num; ++i) {
        machine_id2sort_order.at(i) = (i + total_machine_num - machine_id - 1) % total_machine_num;
      }
      std::sort(commnet_nodes_with_sort_val.begin(), commnet_nodes_with_sort_val.end(),
                [&](const std::pair<CopyCommNetTaskNode*, int64_t>& lhs,
                    const std::pair<CopyCommNetTaskNode*, int64_t>& rhs) {
                  if (lhs.first->peer_machine_id() == rhs.first->peer_machine_id()) {
                    return lhs.second < rhs.second;
                  }
                  return machine_id2sort_order.at(lhs.first->peer_machine_id())
                         < machine_id2sort_order.at(rhs.first->peer_machine_id());
                });

      for (size_t i = 0; i < commnet_nodes_with_sort_val.size() - 1; ++i) {
        commnet_nodes_with_sort_val.at(i).first->BuildCtrlRegstDescIfNeed(
            commnet_nodes_with_sort_val.at(i + 1).first);
      }
    }
  }
}

template<typename TaskNodeType>
void TaskGraph::CollectCopyCommNetForReduceTaskNodes(
    const std::vector<TaskNodeType*>& reduce_task_nodes,
    std::vector<std::pair<CopyCommNetTaskNode*, int64_t>>* commnet_nodes_with_sort_val) {
  HashSet<CopyCommNetTaskNode*> inserted_commnet_nodes;
  for (TaskNodeType* reduce_task_node : reduce_task_nodes) {
    for (TaskEdge* in_edge : reduce_task_node->in_edges()) {
      TaskNode* pre_node = in_edge->src_node();

      while (IsEndingTaskType<TaskNodeType>(pre_node->GetTaskType()) == false) {
        if (pre_node->GetTaskType() == TaskType::kCopyCommNet) {
          CopyCommNetTaskNode* commnet_node = dynamic_cast<CopyCommNetTaskNode*>(pre_node);
          CHECK(commnet_node != nullptr);

          if (inserted_commnet_nodes.find(commnet_node) == inserted_commnet_nodes.end()) {
            commnet_nodes_with_sort_val->emplace_back(
                commnet_node, reduce_task_node->parallel_ctx()->parallel_id());
            inserted_commnet_nodes.insert(commnet_node);
          }
          break;
        }
        pre_node = pre_node->SoleInEdge()->src_node();
      }
    }
  }
}

template<>
bool TaskGraph::IsEndingTaskType<ReduceGlobalAddCompTaskNode>(TaskType type) {
  return type == TaskType::kReduceLocalAdd;
}

template<>
bool TaskGraph::IsEndingTaskType<ReduceGatherCompTaskNode>(TaskType type) {
  return type == TaskType::kReduceGlobalAdd;
}

void TaskGraph::AddMutexCtrlEdgeInSameChain() { UNIMPLEMENTED(); }

void TaskGraph::RmUselessConsumeRelationshipBetweenFwBw() {
  for (TaskNode* task_node : ordered_task_nodes_) {
    auto bw_node = dynamic_cast<NormalBackwardCompTaskNode*>(task_node);
    if (bw_node == nullptr) { continue; }
    bw_node->RmUselessConsumeRelationshipToFw();
  }
}

void TaskGraph::AddOrderCtrlEdgeBetweenCopyAndMdUpdt() {
  for (TaskNode* task_node : ordered_task_nodes_) {
    auto copy_hd_task_node = dynamic_cast<CopyHdTaskNode*>(task_node);
    if (copy_hd_task_node == nullptr) { continue; }
    if (copy_hd_task_node->copy_type() != CopyHdOpConf::H2D) { continue; }
    if (copy_hd_task_node->area_id() != static_cast<int64_t>(kDataForwardArea)
        && copy_hd_task_node->area_id() != static_cast<int64_t>(kBoundaryArea)) {
      continue;
    }
    std::vector<TaskNode*> candidate_nodes;
    auto ForEachNextNode = [&](TaskNode* node,
                               const std::function<void(TaskNode*)>& TryPushNodeToQueue) {
      auto fw_task_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
      if (fw_task_node != nullptr && fw_task_node->logical_node()->HasOpWithModelBlob()) { return; }
      node->ForEachNodeOnOutEdge([&](TaskNode* node_on_out_edge) {
        if (IsForwardTaskType(node_on_out_edge->GetTaskType())) {
          TryPushNodeToQueue(node_on_out_edge);
        }
      });
    };
    auto HandlerAddCandidate = [&](TaskNode* node) {
      auto fw_task_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
      if (fw_task_node != nullptr && fw_task_node->logical_node()->HasOpWithModelBlob()
          && fw_task_node->parallel_ctx()->parallel_num() > 1
          && fw_task_node->parallel_ctx()->policy() == kDataParallel) {
        candidate_nodes.push_back(node);
      }
    };
    BfsForEachNode({task_node}, ForEachNextNode, HandlerAddCandidate);
    std::sort(candidate_nodes.begin(), candidate_nodes.end(),
              [](const TaskNode* a, const TaskNode* b) {
                return a->order_in_graph() < b->order_in_graph();
              });
    int64_t last_chain_id = -1;
    for (TaskNode* candidate_node : candidate_nodes) {
      if (candidate_node->chain_id() != last_chain_id) {
        last_chain_id = candidate_node->chain_id();
        candidate_node->ForEachNodeOnInEdge([&](TaskNode* node_on_in_edge) {
          if (IsMdUpdtTaskType(node_on_in_edge->GetTaskType())) {
            RegstDesc* ctrl_regst = task_node->BuildCtrlRegstDesc(node_on_in_edge);
            RegstDesc* copy_out_regst = copy_hd_task_node->GetProducedRegst("copy_out").get();
            int64_t piece_num_in_batch = Global<JobDesc>::Get()->NumOfPiecesInBatch();
            ctrl_regst->UpdtMinRegstNumIfNeed(copy_out_regst->min_register_num()
                                              + piece_num_in_batch - 1);
            CtrlRegstDesc* ctrl_regst_desc =
                ctrl_regst->mut_regst_desc_type()->mutable_ctrl_regst_desc();
            ctrl_regst_desc->set_reliant_regst_desc_id(copy_out_regst->regst_desc_id());
            ctrl_regst_desc->set_returned_regst_num(piece_num_in_batch);
          }
        });
      }
    }
  }
}

void TaskGraph::SetAreaIdForNewNodes(const LogicalNode* src_logical,
                                     const LogicalNode* dst_logical) {
  CHECK(src_logical != nullptr && dst_logical != nullptr);
  ForEachNode([&](TaskNode* node) {
    if (node->area_id() != static_cast<int64_t>(kInvalidArea)) return;
    if (src_logical->GetAreaId() == dst_logical->GetAreaId()) {
      node->set_area_id(src_logical->GetAreaId());
    } else {
      node->set_area_id(static_cast<int64_t>(kBoundaryArea));
    }
  });
}

#define DEFINE_BLD_SUB_TASK_GRAPH_METHOD(method_name) \
  void TaskGraph::method_name BLD_SUB_TSK_GPH_MTHD_ARGS()

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxing) {
  std::vector<TaskNode*>* sorted_out_box = nullptr;
  if (logical2sorted_out_box->find(src_logical) == logical2sorted_out_box->end()) {
    BuildOutBoxing(src_logical, sorted_src_comp_tasks, &((*logical2sorted_out_box)[src_logical]),
                   MutBufTask, AllocateCpuThrdIdEvenly);
  }
  sorted_out_box = &(logical2sorted_out_box->at(src_logical));

  std::vector<TaskNode*>* sorted_in_box = nullptr;
  if (logical2sorted_in_box->find(dst_logical) == logical2sorted_in_box->end()) {
    BuildInBoxing(dst_logical, sorted_dst_comp_tasks, &((*logical2sorted_in_box)[dst_logical]),
                  AllocateCpuThrdIdEvenly);
  }
  sorted_in_box = &(logical2sorted_in_box->at(dst_logical));

  for (TaskNode* src_box : *sorted_out_box) {
    for (TaskNode* dst_box : *sorted_in_box) { ConnectWithCopyCommNetIfNeed(src_box, dst_box); }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    CompTaskNode* src = sorted_src_comp_tasks[i];
    CompTaskNode* dst = sorted_dst_comp_tasks[i];
    BuildTaskPath(src, dst, MutBufTask, true);
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphBySelectOneSourceToSoleSink) {
  CHECK_EQ(sorted_dst_comp_tasks.size(), 1);
  CompTaskNode* sole_dst_comp_task = sorted_dst_comp_tasks.front();
  CompTaskNode* selected_src_comp_task = nullptr;
  bool is_same_machine = false;
  auto UpdateSelected = [&](CompTaskNode* node) {
    selected_src_comp_task = node;
    is_same_machine = (node->machine_id() == sole_dst_comp_task->machine_id());
  };
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    if (selected_src_comp_task == nullptr) {
      UpdateSelected(src_comp_task);
      continue;
    }
    if (src_comp_task->machine_id() == sole_dst_comp_task->machine_id()) {
      if (is_same_machine == false) {
        UpdateSelected(src_comp_task);
        continue;
      }
      if (src_comp_task->thrd_id() == sole_dst_comp_task->thrd_id()) {
        UpdateSelected(src_comp_task);
        break;
      }
    }
  }
  CHECK_NOTNULL(selected_src_comp_task);
  BldSubTskGphByOneToOne(nullptr, nullptr, {selected_src_comp_task}, sorted_dst_comp_tasks, nullptr,
                         nullptr, MutBufTask, AllocateCpuThrdIdEvenly);
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceScatter2ReduceLocalAdd) {
  int64_t edge_duplicate_num =
      sorted_src_comp_tasks.front()->logical_node()->parallel_desc()->sorted_machine_ids().size();
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      if (src_comp_task->machine_id() == dst_comp_task->machine_id()) {
        for (int64_t i = 0; i < edge_duplicate_num; ++i) {
          BuildTaskPath(src_comp_task, dst_comp_task, MutBufTask, false);
        }
      }
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceScatter2ReduceGlobalAdd) {
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      BuildTaskPath(src_comp_task, dst_comp_task, MutBufTask, false);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceLocalAdd2ReduceGlobalAdd) {
  int64_t device_num_of_each_machine =
      sorted_src_comp_tasks.front()->logical_node()->parallel_desc()->device_num_of_each_machine();
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    int64_t src_id_inside_the_machine = src_comp_task->parallel_id() % device_num_of_each_machine;
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      int64_t dst_id_inside_the_machine = dst_comp_task->parallel_id() % device_num_of_each_machine;
      if (src_id_inside_the_machine == dst_id_inside_the_machine) {
        BuildTaskPath(src_comp_task, dst_comp_task, MutBufTask, false);
      }
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceGlobalAdd2ReduceGather) {
  CHECK_GE(sorted_src_comp_tasks.size(), 2);
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      BuildTaskPath(src_comp_task, dst_comp_task, MutBufTask, true);
    }
  }
}

void TaskGraph::BuildTaskPath(
    CompTaskNode* src, CompTaskNode* dst,
    std::function<TaskNode**(CompTaskNode* src, int64_t machine_id, int32_t mem_zone_id)>
        MutBufTask,
    bool use_buf_task_node) {
  CHECK_NE(src, dst);
  auto GetBufTask = [&](int64_t machine_id, int32_t mem_zone_id) {
    return *MutBufTask(src, machine_id, mem_zone_id);
  };
  auto SetBufTask = [&](int64_t machine_id, int32_t mem_zone_id, TaskNode* new_val) {
    TaskNode** cur_val = MutBufTask(src, machine_id, mem_zone_id);
    if (*cur_val == nullptr) {
      *cur_val = new_val;
    } else {
      CHECK_EQ(*cur_val, new_val);
    }
    return new_val;
  };

  TaskNode* cur_node = src;
  while (cur_node->machine_id() != dst->machine_id()
         || cur_node->MemZoneId121() != dst->MemZoneId121()) {
    cur_node = BuildTaskStep(cur_node, dst, GetBufTask, SetBufTask, use_buf_task_node);
  }
  Connect<TaskNode>(cur_node, NewEdge(), dst);
}

TaskNode* TaskGraph::BuildTaskStep(
    TaskNode* cur_node, TaskNode* dst,
    std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id)> GetBufTask,
    std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id, TaskNode*)> SetBufTask,
    bool use_buf_task_node) {
  int32_t cpu_mem_zone_id = Global<IDMgr>::Get()->CpuMemZoneId();
  int32_t next_mem_zone_id = -1;
  TaskNode* next_node = nullptr;
  if (cur_node->MemZoneId121() != cpu_mem_zone_id) {
    next_mem_zone_id = cpu_mem_zone_id;
    if (!use_buf_task_node || !(next_node = GetBufTask(cur_node->machine_id(), next_mem_zone_id))) {
      next_node = AddCopyD2HTaskFrom(cur_node);
      Connect<TaskNode>(cur_node, NewEdge(), next_node);
    }
  } else if (cur_node->machine_id() == dst->machine_id()) {
    next_mem_zone_id = dst->MemZoneId121();
    if (!use_buf_task_node || !(next_node = GetBufTask(cur_node->machine_id(), next_mem_zone_id))) {
      next_node = AddCopyH2DTaskTo(dst);
      Connect<TaskNode>(cur_node, NewEdge(), next_node);
    }
  } else if (cur_node->machine_id() != dst->machine_id()) {
    next_mem_zone_id = cpu_mem_zone_id;
    if (!use_buf_task_node || !(next_node = GetBufTask(dst->machine_id(), next_mem_zone_id))) {
      next_node = AddCopyCommNetTaskBetween(cur_node, dst);
      Connect<TaskNode>(cur_node, NewEdge(), next_node);
    }
  } else {
    UNIMPLEMENTED();
  }
  if (use_buf_task_node) { SetBufTask(next_node->machine_id(), next_mem_zone_id, next_node); }
  return next_node;
}

TaskNode* TaskGraph::AddCopyH2DTaskTo(TaskNode* task) {
  CHECK_EQ(task->device_type(), DeviceType::kGPU);
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  copy_task->Init(CopyHdOpConf::H2D, task->machine_id(), task->GpuPhyId());
  return copy_task;
}

TaskNode* TaskGraph::AddCopyD2HTaskFrom(TaskNode* task) {
  CHECK_EQ(task->device_type(), DeviceType::kGPU);
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  copy_task->Init(CopyHdOpConf::D2H, task->machine_id(), task->GpuPhyId());
  return copy_task;
}

TaskNode* TaskGraph::AddCopyCommNetTaskBetween(TaskNode* src, TaskNode* dst) {
  CHECK_NE(src->machine_id(), dst->machine_id());
  CopyCommNetTaskNode* copy_comm_net_task = NewNode<CopyCommNetTaskNode>();
  copy_comm_net_task->Init(dst->machine_id(), src->machine_id());
  return copy_comm_net_task;
}

void TaskGraph::BuildOutBoxing(
    const LogicalNode* logical, const std::vector<CompTaskNode*>& sorted_comp_tasks,
    std::vector<TaskNode*>* sorted_out_box,
    std::function<TaskNode**(CompTaskNode* src, int64_t machine_id, int32_t mem_zone_id)>
        MutBufTask,
    std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly) {
  std::map<int64_t, std::vector<TaskNode*>> machine_id2bound_task;
  for (CompTaskNode* comp_task : sorted_comp_tasks) {
    TaskNode* task = comp_task;
    if (task->device_type() == DeviceType::kGPU) {
      TaskNode** buf_task =
          MutBufTask(comp_task, comp_task->machine_id(), Global<IDMgr>::Get()->CpuMemZoneId());
      if ((*buf_task) == nullptr) {
        task = AddCopyD2HTaskFrom(comp_task);
        Connect<TaskNode>(comp_task, NewEdge(), task);
        *buf_task = task;
      } else {
        task = *buf_task;
      }
    }
    machine_id2bound_task[task->machine_id()].push_back(task);
  }
  for (const auto& pair : machine_id2bound_task) {
    OutBoxingTaskNode* boxing_task = NewNode<OutBoxingTaskNode>();
    boxing_task->set_machine_id(pair.second.front()->machine_id());
    boxing_task->set_thrd_id(AllocateCpuThrdIdEvenly(boxing_task));
    for (TaskNode* task : pair.second) { Connect<TaskNode>(task, NewEdge(), boxing_task); }
    sorted_out_box->push_back(boxing_task);
  }
}

void TaskGraph::BuildInBoxing(const LogicalNode* logical,
                              const std::vector<CompTaskNode*>& sorted_comp_tasks,
                              std::vector<TaskNode*>* sorted_in_box,
                              std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly) {
  std::map<int64_t, std::vector<TaskNode*>> machine_id2bound_task;
  for (CompTaskNode* comp_task : sorted_comp_tasks) {
    TaskNode* task = comp_task;
    if (task->device_type() == DeviceType::kGPU) {
      task = AddCopyH2DTaskTo(comp_task);
      Connect<TaskNode>(task, NewEdge(), comp_task);
    }
    machine_id2bound_task[task->machine_id()].push_back(task);
  }
  for (const auto& pair : machine_id2bound_task) {
    InBoxingTaskNode* boxing_task = NewNode<InBoxingTaskNode>();
    boxing_task->set_machine_id(pair.second.front()->machine_id());
    boxing_task->set_thrd_id(AllocateCpuThrdIdEvenly(boxing_task));
    for (TaskNode* task : pair.second) { Connect<TaskNode>(boxing_task, NewEdge(), task); }
    sorted_in_box->push_back(boxing_task);
  }
}

void TaskGraph::ConnectWithCopyCommNetIfNeed(TaskNode* src, TaskNode* dst) {
  if (src->machine_id() == dst->machine_id()) {
    Connect(src, NewEdge(), dst);
  } else {
    TaskNode* copy_comm_net_task = AddCopyCommNetTaskBetween(src, dst);
    Connect<TaskNode>(src, NewEdge(), copy_comm_net_task);
    Connect<TaskNode>(copy_comm_net_task, NewEdge(), dst);
  }
}

bool IsBackEdge(TaskNode* src, TaskNode* dst) {
  return src->GetTaskType() == TaskType::kNormalMdUpdt
         && (dst->GetTaskType() == TaskType::kNormalForward
             || dst->GetTaskType() == TaskType::kNormalBackward);
}

}  // namespace oneflow
