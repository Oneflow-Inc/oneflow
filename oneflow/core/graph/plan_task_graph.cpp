#include "oneflow/core/graph/plan_task_graph.h"

namespace oneflow {

bool PlanTaskGraph::IsReachableInSameArea(int64_t src_task_id, int64_t dst_task_id) const {
  return IsReachableToAncestor(task_id2plan_task_node_.at(dst_task_id),
                               task_id2plan_task_node_.at(src_task_id));
}

PlanTaskGraph::PlanTaskGraph(const Plan& plan) {
  InitNodes(plan);
  InitEdges();
  InitNode2Ancestor();
}

void PlanTaskGraph::InitNodes(const Plan& plan) {
  for (const auto& task : plan.task()) {
    PlanTaskNode* plan_task_node = new PlanTaskNode(task);
    task_id2plan_task_node_.insert({task.task_id(), plan_task_node});
    AddAllocatedNode(plan_task_node);
  }
}

void PlanTaskGraph::InitEdges() {
  for (const auto& task_id_and_plan_task_node : task_id2plan_task_node_) {
    PlanTaskNode* producer_node = task_id_and_plan_task_node.second;
    for (const auto& pair : producer_node->task_proto()->produced_regst_desc()) {
      for (int64_t consumer_task_id : pair.second.consumer_task_id()) {
        PlanTaskNode* consumer_node = task_id2plan_task_node_.at(consumer_task_id);
        if (producer_node->area_id() == consumer_node->area_id()) {
          Connect(producer_node, NewEdge(), consumer_node);
        }
      }
    }
  }
}

void PlanTaskGraph::InitNode2Ancestor() {
  TopoForEachNode([&](const PlanTaskNode* node) {
    node->ForEachNodeOnInEdge([&](const PlanTaskNode* prev) {
      node2ancestors_[node].insert(prev);
      node2ancestors_[node].insert(node2ancestors_[prev].begin(), node2ancestors_[prev].end());
    });
  });
}

bool PlanTaskGraph::IsAnyNodeReachableToAncestor(const HashSet<const PlanTaskNode*>& nodes,
                                                 const PlanTaskNode* ancestor) const {
  for (const PlanTaskNode* node : nodes) {
    if (node2ancestors_.at(node).find(ancestor) != node2ancestors_.at(node).end()) { return true; }
  }
  return false;
}

bool PlanTaskGraph::IsReachableToAncestor(const PlanTaskNode* node,
                                          const PlanTaskNode* ancestor) const {
  return node2ancestors_.at(node).find(ancestor) != node2ancestors_.at(node).end();
}

void PlanTaskGraph::SortByProducerTaskTopoOrder(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<void(const RegstDescProto*)>& Handler) const {
  HashMap<const PlanTaskNode*, const RegstDescProto*> producer_node2regst_desc;
  for (const auto* regst_desc : regst_descs) {
    const auto* producer = task_id2plan_task_node_.at(regst_desc->producer_task_id());
    CHECK(producer_node2regst_desc.emplace(producer, regst_desc).second);
  }
  auto IsSourceNode = [&](const PlanTaskNode* node) {
    const auto& ancestors = node2ancestors_.at(node);
    for (const auto& pair : producer_node2regst_desc) {
      if (ancestors.find(pair.first) != ancestors.end()) { return false; }
    }
    return true;
  };
  auto IsSinkNode = [&](const PlanTaskNode* node) {
    for (const auto& pair : producer_node2regst_desc) {
      const auto& ancestors = node2ancestors_.at(pair.first);
      if (ancestors.find(node) != ancestors.end()) { return false; }
    }
    return true;
  };
  const PlanTaskNode* start_node = nullptr;
  const PlanTaskNode* end_node = nullptr;
  for (const auto& pair : producer_node2regst_desc) {
    if (IsSourceNode(pair.first)) {
      CHECK(start_node == nullptr);
      start_node = pair.first;
    }
    if (IsSinkNode(pair.first)) {
      CHECK(end_node == nullptr);
      end_node = pair.first;
    }
  }
  auto ForEachInNode = [&](const PlanTaskNode* node,
                           const std::function<void(const PlanTaskNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](const PlanTaskNode* in_node) {
      if (in_node == start_node || IsReachableToAncestor(in_node, start_node)) { Handler(in_node); }
    });
  };
  auto ForEachOutNode = [&](const PlanTaskNode* node,
                            const std::function<void(const PlanTaskNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](const PlanTaskNode* out_node) {
      if (out_node == end_node || IsReachableToAncestor(end_node, out_node)) { Handler(out_node); }
    });
  };
  TopoForEachNode({start_node}, ForEachInNode, ForEachOutNode,
                  [&](const PlanTaskNode* node) { Handler(producer_node2regst_desc.at(node)); });
}

void PlanTaskGraph::ComputeLifetimeSameStreamActorIds(
    const RegstDescProto* regst_desc, HashSet<int64_t>* lifetime_same_stream_actor_ids) const {
  int64_t stream_id =
      Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(regst_desc->producer_task_id());
  ComputeLifetimeActorIds(regst_desc, lifetime_same_stream_actor_ids, [&](int64_t task_id) {
    return Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(task_id) == stream_id;
  });
}

void PlanTaskGraph::ComputeLifetimeSameChainActorIds(
    const RegstDescProto* regst_desc, HashSet<int64_t>* lifetime_same_chain_actor_ids) const {
  int64_t chain_id = task_id2plan_task_node_.at(regst_desc->producer_task_id())->chain_id();
  ComputeLifetimeActorIds(regst_desc, lifetime_same_chain_actor_ids, [&](int64_t task_id) {
    return task_id2plan_task_node_.at(task_id)->chain_id() == chain_id;
  });
}

void PlanTaskGraph::ComputeLifetimeActorIds(const RegstDescProto* regst_desc,
                                            HashSet<int64_t>* lifetime_actor_ids,
                                            const std::function<bool(int64_t)>& IsAllowed) const {
  const auto* producer = task_id2plan_task_node_.at(regst_desc->producer_task_id());
  HashSet<const PlanTaskNode*> consumers;
  for (int64_t consumer_task_id : regst_desc->consumer_task_id()) {
    consumers.insert(task_id2plan_task_node_.at(consumer_task_id));
  }
  auto ForEachInNode = [&](const PlanTaskNode* node,
                           const std::function<void(const PlanTaskNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](const PlanTaskNode* prev) {
      if (prev == producer || IsReachableToAncestor(prev, producer)) { Handler(prev); }
    });
  };
  auto ForEachOutNode = [&](const PlanTaskNode* node,
                            const std::function<void(const PlanTaskNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](const PlanTaskNode* next) {
      if (consumers.find(next) != consumers.end()
          || IsAnyNodeReachableToAncestor(consumers, next)) {
        Handler(next);
      }
    });
  };
  TopoForEachNode({producer}, ForEachInNode, ForEachOutNode, [&](const PlanTaskNode* node) {
    if (IsAllowed(node->task_id())) { lifetime_actor_ids->insert(node->task_id()); }
  });
}

void PlanTaskGraph::AssertThereIsOnlyOneTopoOrder(
    const HashSet<int64_t>& same_stream_node_ids) const {
  auto ForEachInNode = [&](const PlanTaskNode* node,
                           const std::function<void(const PlanTaskNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](const PlanTaskNode* in_node) {
      if (same_stream_node_ids.find(in_node->task_id()) != same_stream_node_ids.end()) {
        Handler(in_node);
      }
    });
  };
  auto ForEachOutNode = [&](const PlanTaskNode* node,
                            const std::function<void(const PlanTaskNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](const PlanTaskNode* out_node) {
      if (same_stream_node_ids.find(out_node->task_id()) != same_stream_node_ids.end()) {
        Handler(out_node);
      }
    });
  };
  HashMap<const PlanTaskNode*, int> node2depth;
  std::list<const PlanTaskNode*> starts;
  for (int64_t task_id : same_stream_node_ids) {
    int in_num = 0;
    const PlanTaskNode* node = task_id2plan_task_node_.at(task_id);
    ForEachInNode(node, [&](const PlanTaskNode* in_node) { ++in_num; });
    if (in_num == 0) { starts.push_back(node); }
  }
  TopoForEachNode(starts, ForEachInNode, ForEachOutNode, [&](const PlanTaskNode* node) {
    int in_nodes_max_depth = -1;
    ForEachInNode(node, [&](const PlanTaskNode* in_node) {
      in_nodes_max_depth = std::max(in_nodes_max_depth, node2depth.at(in_node));
    });
    node2depth[node] = in_nodes_max_depth + 1;
  });
  HashMap<int, size_t> depth2same_depth_node_num;
  for (const auto& pair : node2depth) {
    ++depth2same_depth_node_num[pair.second];
    CHECK_EQ(depth2same_depth_node_num.at(pair.second), 1);
  }
}

}  // namespace oneflow
