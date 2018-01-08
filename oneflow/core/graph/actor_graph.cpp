#include "oneflow/core/graph/actor_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

namespace {

std::unordered_set<ActorEdge*> FindBackEdges(const ActorGraph& actor_graph) {
  TODO();
  return std::unordered_set<ActorEdge*>();
}

class UnfoldedActorNode;
class UnfoldedActorEdge final
    : public Edge<UnfoldedActorNode, UnfoldedActorEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnfoldedActorEdge);
  UnfoldedActorEdge() = default;
  ~UnfoldedActorEdge() = default;
};

class UnfoldedActorNode final
    : public Node<UnfoldedActorNode, UnfoldedActorEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnfoldedActorNode);
  explicit UnfoldedActorNode(const ActorNode* actor_node, int64_t act_id,
                             double time_per_act)
      : actor_node_(actor_node), act_id_(act_id), time_per_act_(time_per_act) {}
  ~UnfoldedActorNode() = default;

  // Getters
  const ActorNode* actor_node() const { return actor_node_; }

 private:
  const ActorNode* actor_node_;
  int64_t act_id_;
  double time_per_act_;
};

class UnfoldedActorGraph final
    : public Graph<UnfoldedActorNode, UnfoldedActorEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnfoldedActorGraph);
  UnfoldedActorGraph(
      const ActorGraph& actor_graph,
      const HashMap<int64_t, std::list<const ActEvent*>>& task_id2act_evts);
  ~UnfoldedActorGraph() = default;
  double CalcActorNodeAvgLongestPathTime(
      const ActorNode* start_actor_node,
      const std::unordered_set<const ActorNode*>& end_actor_nodes) const;

 private:
  double CalcLongestPathTime(
      const UnfoldedActorNode* start_node,
      const std::list<const UnfoldedActorNode*>& end_nodes) const;
  void CreateNodes(
      const HashMap<int64_t, std::list<const ActEvent*>>& task_id2act_evts);
  void ConnectNodes(std::list<UnfoldedActorNode*>* src_nodes,
                    std::list<UnfoldedActorNode*>* dst_nodes, bool is_back_edge,
                    TaskType src_task_type);
  const ActorGraph* actor_graph_;
  HashMap<const ActorNode*, std::list<UnfoldedActorNode*>> actor_node2unfolded_;
};

std::list<const UnfoldedActorNode*> FilterNodesOnOutEdgeByActorNode(
    const UnfoldedActorNode* node,
    const std::unordered_set<const ActorNode*>& actor_nodes) {
  std::list<const UnfoldedActorNode*> filtered_nodes;
  node->ForEachNodeOnOutEdge([&](UnfoldedActorNode* next) {
    if (actor_nodes.find(next->actor_node()) != actor_nodes.end()) {
      filtered_nodes.push_back(next);
    }
  });
  return filtered_nodes;
}

double UnfoldedActorGraph::CalcActorNodeAvgLongestPathTime(
    const ActorNode* start_actor_node,
    const std::unordered_set<const ActorNode*>& end_actor_nodes) const {
  const auto& unfolded_nodes = actor_node2unfolded_.at(start_actor_node);
  CHECK(!unfolded_nodes.empty());
  double sum = 0;
  for (const UnfoldedActorNode* node : unfolded_nodes) {
    auto filtered_unfolded_end_nodes =
        FilterNodesOnOutEdgeByActorNode(node, end_actor_nodes);
    CHECK(!filtered_unfolded_end_nodes.empty());
    sum += CalcLongestPathTime(node, filtered_unfolded_end_nodes);
  }
  return sum / unfolded_nodes.size();
}

double UnfoldedActorGraph::CalcLongestPathTime(
    const UnfoldedActorNode* start_node,
    const std::list<const UnfoldedActorNode*>& end_nodes) const {
  TODO();
  return 0;
}

HashMap<int64_t, std::list<const ActEvent*>> TakingActEventsOfTwoPieces(
    const HashMap<int64_t, std::list<const ActEvent*>>& task_id2act_evts) {
  TODO();
  return task_id2act_evts;
}

void UnfoldedActorGraph::CreateNodes(
    const HashMap<int64_t, std::list<const ActEvent*>>& task_id2act_evts) {
  TODO();
  actor_node2unfolded_ = {};
}

void UnfoldedActorGraph::ConnectNodes(std::list<UnfoldedActorNode*>* src_nodes,
                                      std::list<UnfoldedActorNode*>* dst_nodes,
                                      bool is_back_edge,
                                      TaskType src_task_type) {
  TODO();
}

UnfoldedActorGraph::UnfoldedActorGraph(
    const ActorGraph& actor_graph,
    const HashMap<int64_t, std::list<const ActEvent*>>& task_id2act_evts)
    : actor_graph_(&actor_graph) {
  auto back_edges = FindBackEdges(actor_graph);
  CreateNodes(TakingActEventsOfTwoPieces(task_id2act_evts));
  actor_graph.ForEachEdge([&](ActorEdge* actor_edge) {
    ConnectNodes(&actor_node2unfolded_.at(actor_edge->src_node()),
                 &actor_node2unfolded_.at(actor_edge->dst_node()),
                 back_edges.find(actor_edge) != back_edges.end(),
                 actor_edge->src_node()->task_type());
  });
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
