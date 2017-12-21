#include "oneflow/core/graph/actor_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

void ActorGraph::MakeRegstDescId2AvgLifeTimeHash(
    std::unordered_map<uint64_t, double>* regst_desc_id2life_time,
    const std::function<double(uint64_t)>& AvgDuration4TaskId) const {
  // TODO
}

ActorGraph::ActorGraph(const Plan& plan,
                       std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  // TODO
}

void ActorGraph::MakeTaskId2AvgDurationHash(
    std::unordered_map<uint64_t, double>* task_id2avg_duration) const {
  // TODO
}

double ActorGraph::InitiationInterval() const {
  // TODO
  return 0;
}

}  // namespace oneflow
