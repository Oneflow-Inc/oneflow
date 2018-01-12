#include "oneflow/core/graph/actor_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

namespace {

std::string GenActUid(int64_t actor_id, int64_t act_id) {
  return std::to_string(actor_id) + ":" + std::to_string(act_id);
}

class ActNode;
class ActEdge final : public Edge<ActNode, ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActEdge);
  ActEdge() = default;
  ~ActEdge() = default;
};

class ActNode final : public Node<ActNode, ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActNode);
  explicit ActNode(const ActEvent& act_event) : act_event_(act_event) {}
  ~ActNode() = default;

  // Getters
  int64_t actor_id() const { return act_event_.actor_id(); }

 private:
  ActEvent act_event_;
};

class ActGraph final : public Graph<ActNode, ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActGraph);
  explicit ActGraph(const std::list<ActEvent>& task_id2act_evts);
  ~ActGraph() = default;
  double CalcActorNodesLongestPathTime(
      int64_t start_actor_ids,
      const std::unordered_set<int64_t>& end_actor_ids) const;

 private:
  double CalcLongestPathTime(const ActNode* start_node,
                             const std::list<const ActNode*>& end_nodes) const;
  void CreateNodes(const std::list<ActEvent>& task_id2act_evts);
  void ConnectNodes();
  HashMap<int64_t, std::list<ActNode*>> actor_id2act_nodes_;
  HashMap<std::string, ActNode*> act_uid2act_node_;
};

std::list<const ActNode*> FilterNodesOnOutEdgeByActorId(
    const ActNode* node, const std::unordered_set<int64_t>& actor_ids) {
  std::list<const ActNode*> filtered_nodes;
  node->ForEachNodeOnOutEdge([&](ActNode* next) {
    if (actor_ids.find(next->actor_id()) != actor_ids.end()) {
      filtered_nodes.push_back(next);
    }
  });
  return filtered_nodes;
}

double ActGraph::CalcActorNodesLongestPathTime(
    int64_t start_actor_id,
    const std::unordered_set<int64_t>& end_actor_ids) const {
  double sum = 0;
  int end_node_num = 0;
  for (const auto* node : actor_id2act_nodes_.at(start_actor_id)) {
    auto related_act_nodes = FilterNodesOnOutEdgeByActorId(node, end_actor_ids);
    if (!related_act_nodes.empty()) {
      sum += CalcLongestPathTime(node, related_act_nodes);
      ++end_node_num;
    }
  }
  CHECK(end_node_num != 0);
  return sum / end_node_num;
}

double ActGraph::CalcLongestPathTime(
    const ActNode* start_node,
    const std::list<const ActNode*>& end_nodes) const {
  TODO();
  return 0;
}

void ActGraph::CreateNodes(const std::list<ActEvent>& task_id2act_evts) {
  TODO();
  actor_id2act_nodes_ = {};
  act_uid2act_node_ = {};
}

void ActGraph::ConnectNodes() { TODO(); }

ActGraph::ActGraph(const std::list<ActEvent>& task_id2act_evts) {
  CreateNodes(task_id2act_evts);
  ConnectNodes();
}

}  // namespace

ActorGraph::ActorGraph(const Plan& plan,
                       std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  TODO();
}

void ActorGraph::MakeRegstDescId2AvgLifeTimeHash(
    HashMap<int64_t, double>* regst_desc_id2life_time,
    const std::function<double(int64_t)>& AvgDuration4TaskId) const {
  TODO();
}

void ActorGraph::MakeTaskId2AvgDurationHash(
    HashMap<int64_t, double>* task_id2avg_duration) const {
  TODO();
}

double ActorGraph::InitiationInterval() const {
  TODO();
  return 0;
}

}  // namespace oneflow
