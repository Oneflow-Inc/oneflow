#include "oneflow/core/graph/act_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

namespace {

std::string GenRegstUid(int64_t regst_desc_id, int64_t producer_act_id) {
  return std::to_string(regst_desc_id) + ":" + std::to_string(producer_act_id);
}

}  // namespace

std::string ActNode::VisualStr() const {
  TODO();
  return "";
}

void ActNode::ForEachProducedRegstDescId(
    const std::function<void(int64_t)>& Handler) const {
  for (const auto& pair : task_proto_->produced_regst_desc()) {
    Handler(pair.second.regst_desc_id());
  }
}

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

void ActGraph::InitNodes(HashMap<std::string, ActNode*>* regst_uid2producer) {
  HashMap<int64_t, const TaskProto*> actor_id2task_proto;
  for (const TaskProto& task : plan().task()) {
    actor_id2task_proto[task.task_id()] = &task;
  }
  HashMap<std::string, const ActNode*> regst_uid2producer_node;
  for (const ActEvent& act_event : *act_events_) {
    int64_t actor_id = act_event.actor_id();
    int64_t act_id = act_event.act_id();
    ActNode* act_node =
        new ActNode(&act_event, actor_id2task_proto.at(actor_id));
    AddAllocatedNode(act_node);
    act_node->ForEachProducedRegstDescId([&](int64_t regst_desc_id) {
      const auto& regst_uid = GenRegstUid(regst_desc_id, act_id);
      regst_uid2producer->insert({regst_uid, act_node});
    });
  }
}

void ActGraph::InitEdges(
    const HashMap<std::string, ActNode*>& regst_uid2producer) {
  ForEachNode([&](ActNode* node) {
    for (const auto& readable : node->act_event().readable_regst_infos()) {
      const auto& regst_uid =
          GenRegstUid(readable.regst_desc_id(), readable.act_id());
      ActNode* producer = regst_uid2producer.at(regst_uid);
      Connect(producer, NewEdge(), node);
      regst_uid2consumer_acts_[regst_uid].push_back(node);
    }
  });
}

void ActGraph::InitConnectedSubGraphSources() { TODO(); }

ActGraph::ActGraph(const Plan& plan,
                   std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  HashMap<std::string, ActNode*> regst_uid2producer;
  InitNodes(&regst_uid2producer);
  InitEdges(regst_uid2producer);
  InitConnectedSubGraphSources();
}

void ActGraph::ToDotFiles(const std::string& dir) const { TODO(); }

}  // namespace oneflow
