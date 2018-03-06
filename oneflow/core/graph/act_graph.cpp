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

void ActGraph::ForEachRegstDescMeanDuration(
    const std::function<void(int64_t, double)>& Handler) const {
  HashMap<int64_t, double> regst_desc_id2duration;
  HashMap<int64_t, int> regst_desc_id2cnt;
  ForEachRegstUidDuration([&](int64_t regst_desc_id, int64_t, double time) {
    regst_desc_id2duration[regst_desc_id] += time;
    ++regst_desc_id2cnt[regst_desc_id];
  });
  for (const auto& pair : regst_desc_id2duration) {
    Handler(pair.first, pair.second / regst_desc_id2cnt.at(pair.first));
  }
}

void ActGraph::ForEachRegstDescIIScale(
    const std::function<void(int64_t, double)>& Handler) const {
  TODO();
}

void ActGraph::ForEachRegstUidDuration(
    const std::function<void(int64_t regst_desc_id, int64_t act_id,
                             double time)>& Handler) const {
  TODO();
}

ActGraph::ActGraph(const Plan& plan,
                   std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  TODO();
}

void ActGraph::ToDotFiles(const std::string& dir) const { TODO(); }

}  // namespace oneflow
