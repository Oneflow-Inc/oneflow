#include "oneflow/core/graph/act_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

void ActGraph::ForEachRegstLifeTime(
    const std::function<void(int64_t, double)>& Handler) const {
  TODO();
}

double ActGraph::CalcLongestPathTime(
    const ActNode* start_node,
    const std::list<const ActNode*>& end_nodes) const {
  TODO();
  return 0;
}

void ActGraph::CreateNodes() {
  TODO();
  actor_id2act_nodes_ = {};
  act_uid2act_node_ = {};
}

void ActGraph::ConnectNodes() { TODO(); }

ActGraph::ActGraph(const Plan& plan,
                   std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  CreateNodes();
  ConnectNodes();
}

}  // namespace oneflow
