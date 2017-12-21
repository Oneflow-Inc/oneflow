#include "oneflow/core/graph/actor_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/common/dfs_visitor.h"

namespace oneflow {

ActorGraph::ActorGraph(const Plan& plan,
                       std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  // TODO
}

void ActorGraph::DfsForEachNode(
    const std::function<void(ActorNode*)>& Handler) const {
  auto ForEachNext = [](ActorNode* node,
                        const std::function<void(ActorNode*)>& Handler) {
    node->ForEachNodeOnOutEdge(Handler);
  };
  DfsVisitor<ActorNode*> dfs(ForEachNext);
  std::list<ActorNode*> source_nodes;
  ForEachNode([&](ActorNode* node) {
    if (node->in_edges().empty()) { source_nodes.push_back(node); }
  });
  dfs(source_nodes, Handler);
}

void ActorGraph::UpdateDfsOrderValue() {
  uint32_t order_value = 0;
  DfsForEachNode(
      [&](ActorNode* node) { node->set_dfs_order_value(++order_value); });
}

void ActorGraph::MakeRegstDescId2AvgLifeTimeHash(
    HashMap<uint64_t, double>* regst_desc_id2life_time,
    const std::function<double(uint64_t)>& AvgDuration4TaskId) const {
  // TODO
}

void ActorGraph::MakeTaskId2AvgDurationHash(
    HashMap<uint64_t, double>* task_id2avg_duration) const {
  // TODO
}

double ActorGraph::InitiationInterval() const {
  // TODO
  return 0;
}

}  // namespace oneflow
