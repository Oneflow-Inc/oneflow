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

void ActGraph::ForEachRegstUidLifeTime(
    const std::function<void(double, int64_t, int64_t)>& Handler) const {
  for (const auto& sources : connected_subgraph_sources_) {
    ForEachSubGraphRegstUidLifeTime(sources, Handler);
  }
}

void ActGraph::ForEachSubGraphRegstUidLifeTime(
    const std::list<const ActNode*>& sources,
    const std::function<void(double, int64_t, int64_t)>& Handler) const {
  TODO();
}

void ActGraph::InitProducerId2RegstDescIds() {
  for (const TaskProto& task : plan().task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      producer_id2regst_desc_ids_[pair.second.regst_desc_id()].push_back(
          pair.second.producer_task_id());
    }
  }
}

void ActGraph::InitNodes() { TODO(); }

void ActGraph::InitEdges() { TODO(); }

void ActGraph::InitConnectedSubGraphSources() { TODO(); }

ActGraph::ActGraph(const Plan& plan,
                   std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  InitProducerId2RegstDescIds();
  InitNodes();
  InitEdges();
  InitConnectedSubGraphSources();
}

}  // namespace oneflow
