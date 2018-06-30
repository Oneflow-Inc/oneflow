#include "oneflow/core/graph/plan_task_graph.h"

namespace oneflow {

int64_t PlanTaskNode::chain_id() const {
  int64_t chain_id = task_proto_->task_set_info().chain_id();
  CHECK_NE(chain_id, -1);
  return chain_id;
}

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
    if (IsReachableToAncestor(node, ancestor)) { return true; }
  }
  return false;
}

bool PlanTaskGraph::IsReachableToAncestor(const PlanTaskNode* node,
                                          const PlanTaskNode* ancestor) const {
  return node2ancestors_.at(node).find(ancestor) != node2ancestors_.at(node).end();
}

void PlanTaskGraph::SortByProducerTaskOrderInGraph(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<void(const RegstDescProto*)>& Handler) const {
  using ProducerAndRegstDesc = std::pair<const PlanTaskNode*, const RegstDescProto*>;
  std::vector<ProducerAndRegstDesc> producer_and_regst_descs;
  for (const auto* regst_desc : regst_descs) {
    const auto* producer = task_id2plan_task_node_.at(regst_desc->producer_task_id());
    producer_and_regst_descs.emplace_back(std::make_pair(producer, regst_desc));
  }
  std::sort(producer_and_regst_descs.begin(), producer_and_regst_descs.end(),
            [&](const ProducerAndRegstDesc& lhs, const ProducerAndRegstDesc& rhs) {
              return lhs.first->order_in_graph() < rhs.first->order_in_graph();
            });
  for (const auto& pair : producer_and_regst_descs) { Handler(pair.second); }
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
    CHECK(consumers.emplace(task_id2plan_task_node_.at(consumer_task_id)).second);
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
    if (IsAllowed(node->task_id())) { CHECK(lifetime_actor_ids->emplace(node->task_id()).second); }
  });
}

void PlanTaskGraph::AssertThereIsOnlyOneTopoOrder(const HashSet<int64_t>& task_ids) const {
  auto ForEachInNode = [&](const PlanTaskNode* node,
                           const std::function<void(const PlanTaskNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](const PlanTaskNode* in_node) {
      if (task_ids.find(in_node->task_id()) != task_ids.end()) { Handler(in_node); }
    });
  };
  auto ForEachOutNode = [&](const PlanTaskNode* node,
                            const std::function<void(const PlanTaskNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](const PlanTaskNode* out_node) {
      if (task_ids.find(out_node->task_id()) != task_ids.end()) { Handler(out_node); }
    });
  };
  HashMap<const PlanTaskNode*, int> node2depth;
  std::list<const PlanTaskNode*> starts;
  for (int64_t task_id : task_ids) {
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
