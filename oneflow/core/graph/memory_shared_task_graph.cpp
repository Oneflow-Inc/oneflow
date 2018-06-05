#include "oneflow/core/graph/memory_shared_task_graph.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

MemSharedTaskGraph::MemSharedTaskGraph(const Plan& plan) : plan_(&plan) {
  InitNodes();
  InitEdges();
  InitNode2Ancestor();
}

void MemSharedTaskGraph::InitNodes() {
  for (const auto& task : plan_->task()) {
    HashSet<bool> is_enable_mem_sharing;
    for (const auto& pair : task.produced_regst_desc()) {
      if (pair.second.consumer_task_id_size() > 0
          && RtRegstDesc(pair.second).packed_blob_desc()->TotalByteSize() > 0) {
        is_enable_mem_sharing.insert(pair.second.enable_mem_sharing());
      }
    }
    CHECK_LE(is_enable_mem_sharing.size(), 1);
    if (is_enable_mem_sharing.size() == 1 && *is_enable_mem_sharing.begin() == true) {
      auto* mem_shared_task_node = new MemSharedTaskNode(task);
      task_id2mem_shared_task_node_.insert({task.task_id(), mem_shared_task_node});
      AddAllocatedNode(mem_shared_task_node);
    }
  }
}

void MemSharedTaskGraph::InitEdges() {
  for (const auto& task_id_and_mem_shared_task_node : task_id2mem_shared_task_node_) {
    auto* producer_node = task_id_and_mem_shared_task_node.second;
    for (const auto& pair : producer_node->task_proto()->produced_regst_desc()) {
      for (int64_t consumer_task_id : pair.second.consumer_task_id()) {
        const auto& mem_shared_task_node_it = task_id2mem_shared_task_node_.find(consumer_task_id);
        if (mem_shared_task_node_it != task_id2mem_shared_task_node_.end()) {
          Connect(producer_node, NewEdge(), mem_shared_task_node_it->second);
        }
      }
    }
  }
}

void MemSharedTaskGraph::InitNode2Ancestor() {
  TopoForEachNode([&](MemSharedTaskNode* node) {
    node->ForEachNodeOnInEdge([&](MemSharedTaskNode* prev) {
      node2ancestor_[node].insert(prev);
      node2ancestor_[node].insert(node2ancestor_[prev].begin(), node2ancestor_[prev].end());
    });
  });
}

bool MemSharedTaskGraph::IsAnyOneReachable(const HashSet<MemSharedTaskNode*>& nodes,
                                           const MemSharedTaskNode* ancestor) const {
  for (const MemSharedTaskNode* node : nodes) {
    if (node2ancestor_.at(node).find(ancestor) != node2ancestor_.at(node).end()) { return true; }
  }
  return false;
}

HashSet<int64_t> MemSharedTaskGraph::ComputeLifeTimeSameStreamTaskIds4RegstDescId(
    const RegstDescProto& regst_desc) const {
  MemSharedTaskNode* producer = task_id2mem_shared_task_node_.at(regst_desc.producer_task_id());
  HashSet<MemSharedTaskNode*> consumers;
  producer->ForEachNodeOnOutEdge([&](MemSharedTaskNode* node) { consumers.insert(node); });
  auto ForEachInNode = [&](MemSharedTaskNode* node,
                           const std::function<void(MemSharedTaskNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](MemSharedTaskNode* prev) {
      if (prev == producer || IsAnyOneReachable({prev}, producer)) { Handler(prev); }
    });
  };
  auto ForEachOutNode = [&](MemSharedTaskNode* node,
                            const std::function<void(MemSharedTaskNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](MemSharedTaskNode* next) {
      if (consumers.find(next) != consumers.end() || IsAnyOneReachable(consumers, next)) {
        Handler(next);
      }
    });
  };
  int64_t global_work_stream_id =
      Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(regst_desc.producer_task_id());
  HashSet<int64_t> life_time_same_stream_task_ids;
  TopoForEachNode({producer}, ForEachInNode, ForEachOutNode, [&](MemSharedTaskNode* node) {
    int64_t task_id = node->task_proto()->task_id();
    if (Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(task_id) == global_work_stream_id) {
      life_time_same_stream_task_ids.insert(task_id);
    }
  });
  return life_time_same_stream_task_ids;
}

}  // namespace oneflow
