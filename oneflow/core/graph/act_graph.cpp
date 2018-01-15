#include "oneflow/core/graph/act_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

namespace {

std::string GenRegstUid(int64_t regst_desc_id, int64_t producer_act_id) {
  return std::to_string(regst_desc_id) + ":" + std::to_string(producer_act_id);
}

}  // namespace

void ActGraph::ForEachRegstDescLifeTime(
    const std::function<void(int64_t, double)>& Handler) const {
  for (const auto& pair : regst_desc_id2producer_act_ids_) {
    Handler(pair.first, CalcRegstDescLifeTime(pair.first, pair.second));
  }
}

double ActGraph::CalcRegstDescLifeTime(int64_t regst_desc_id,
                                       const std::list<int64_t> act_ids) const {
  double life_time = 0;
  double cnt = 0;
  for (int64_t act_id : act_ids) {
    std::string regst_uid = GenRegstUid(regst_desc_id, act_id);
    const auto& consumers_it = regst_uid2comsumer_acts_.find(regst_uid);
    if (consumers_it == regst_uid2comsumer_acts_.end()) { continue; }
    const ActNode* producer = regst_uid2producer_node_.at(regst_uid);
    life_time += CalcLongestPathTime(producer, consumers_it->second);
    ++cnt;
  }
  if (cnt == 0) { return 0; }
  return life_time / cnt;
}

double ActGraph::CalcLongestPathTime(
    const ActNode* start_node,
    const std::list<const ActNode*>& end_nodes) const {
  TODO();
  return 0;
}

void ActGraph::InitNodes() {
  for (const ActEvent& act_event : *act_events_) {
    int64_t actor_id = act_event.actor_id();
    int64_t act_id = act_event.act_id();
    ActNode* act_node = new ActNode(&act_event);
    AddAllocatedNode(act_node);
    for (int64_t regst_desc_id : producer_id2regst_desc_ids_.at(actor_id)) {
      regst_desc_id2producer_act_ids_[regst_desc_id].push_back(act_id);
      const auto& regst_uid = GenRegstUid(regst_desc_id, act_id);
      regst_uid2producer_node_.insert({regst_uid, act_node});
    }
  }
}

void ActGraph::InitEdges() {
  ForEachNode([&](ActNode* node) {
    for (const auto& readable : node->act_event().readable_regst_infos()) {
      const auto& regst_uid =
          GenRegstUid(readable.regst_desc_id(), readable.act_id());
      ActNode* producer = regst_uid2producer_node_.at(regst_uid);
      Connect(producer, NewEdge(), node);
      regst_uid2comsumer_acts_[regst_uid].push_back(node);
    }
  });
}

void ActGraph::InitProducerId2RegstDescIds() {
  for (const TaskProto& task : plan().task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      producer_id2regst_desc_ids_[pair.second.regst_desc_id()].push_back(
          pair.second.producer_task_id());
    }
  }
}

ActGraph::ActGraph(const Plan& plan,
                   std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  InitProducerId2RegstDescIds();
  InitNodes();
  InitEdges();
}

}  // namespace oneflow
