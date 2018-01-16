#include "oneflow/core/graph/act_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

namespace {

std::string GenRegstUid(int64_t regst_desc_id, int64_t producer_act_id) {
  return std::to_string(regst_desc_id) + ":" + std::to_string(producer_act_id);
}

void ForEachSubGraphNode(const std::list<const ActNode*>& sources,
                         const std::function<void(ActNode*)>& Handler) {
  TODO();
}

void InitSubGraphReachables(
    const std::list<const ActNode*>& sources,
    HashMap<const ActNode*, std::unordered_set<const ActNode*>>*
        node2reachables) {
  TODO();
}

double CalcLongestPathTime(
    const std::function<bool(const ActNode* src, const ActNode* dst)>&
        IsReachable,
    const ActNode* start_node, const std::list<const ActNode*>& end_nodes) {
  TODO();
  return 0;
}

}  // namespace

void ActGraph::ForEachRegstDescLifeTime(
    const std::function<void(int64_t, double)>& Handler) const {
  HashMap<int64_t, double> regst_desc_id2total_time;
  HashMap<int64_t, int> regst_desc_id2cnt;
  ForEachRegstUidLifeTime([&](double time, int64_t regst_desc_id, int64_t) {
    regst_desc_id2total_time[regst_desc_id] += time;
    ++regst_desc_id2cnt[regst_desc_id];
  });
  for (const auto& pair : regst_desc_id2total_time) {
    Handler(pair.first, pair.second / regst_desc_id2cnt.at(pair.first));
  }
}

void ActGraph::ForEachConnectedSubGraphSources(
    const std::function<void(const std::list<const ActNode*>& sources)>&
        Handler) const {
  TODO();
}

void ActGraph::ForEachSubGraphRegstUidLifeTime(
    const std::list<const ActNode*>& sources,
    const std::function<void(double, int64_t, int64_t)>& Handler) const {
  HashMap<const ActNode*, std::unordered_set<const ActNode*>> node2reachables;
  InitSubGraphReachables(sources, &node2reachables);
  auto IsReachable = [&](const ActNode* src, const ActNode* dst) {
    const auto& it = node2reachables.find(src);
    if (it == node2reachables.end()) { return false; }
    return it->second.find(dst) != it->second.end();
  };
  ForEachSubGraphNode(sources, [&](ActNode* node) {
    int64_t actor_id = node->actor_id();
    for (int64_t regst_desc_id : producer_id2regst_desc_ids_.at(actor_id)) {
      const auto& regst_uid = GenRegstUid(regst_desc_id, node->act_id());
      const auto& consumers = regst_uid2consumer_acts_.at(regst_uid);
      if (consumers.empty()) { continue; }
      double time = CalcLongestPathTime(IsReachable, node, consumers);
      Handler(time, regst_desc_id, node->act_id());
    }
  });
}

void ActGraph::ForEachRegstUidLifeTime(
    const std::function<void(double, int64_t, int64_t)>& Handler) const {
  ForEachConnectedSubGraphSources(
      [&](const std::list<const ActNode*>& sources) {
        ForEachSubGraphRegstUidLifeTime(sources, Handler);
      });
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
      regst_uid2consumer_acts_[regst_uid].push_back(node);
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
